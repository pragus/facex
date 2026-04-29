/*
 * threadpool.c — Minimal lock-free thread pool.
 *
 * Uses atomic counter for work distribution + WaitOnAddress for sleeping.
 * Workers spin for ~1µs before sleeping to minimize latency for back-to-back tasks.
 */

#include "threadpool.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

/* Atomic operations */
#if defined(__GNUC__) || defined(__clang__)
#define ATOMIC_LOAD(p)       __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(p, v)   __atomic_store_n(p, v, __ATOMIC_RELEASE)
#define ATOMIC_ADD(p, v)     __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL)
#define ATOMIC_CAS(p, e, d)  __atomic_compare_exchange_n(p, e, d, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#else
#include <stdatomic.h>
#define ATOMIC_LOAD(p)       atomic_load(p)
#define ATOMIC_STORE(p, v)   atomic_store(p, v)
#define ATOMIC_ADD(p, v)     atomic_fetch_add(p, v)
#endif

#define MAX_THREADS 16

/* Shared task state */
typedef struct {
    tp_task_fn fn;
    void* ctx;
    volatile int next_idx;    /* next work index to claim (atomic) */
    int total;                /* total work items */
    int grain;                /* items per chunk */
    volatile int done_count;  /* number of workers finished (atomic) */
    int n_workers;            /* number of workers participating */
    volatile int phase;       /* incremented to signal new work */
} TaskState;

static TaskState g_task;

/* Worker threads */
typedef struct {
    int id;
#ifdef _WIN32
    HANDLE handle;
#else
    pthread_t handle;
#endif
    volatile int alive;
} Worker;

static Worker g_workers[MAX_THREADS];
static int g_n_threads = 0;
static volatile int g_shutdown = 0;

static inline void tp_cpu_relax(void) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* Worker function: spin-wait for work, execute chunks, signal done */
static
#ifdef _WIN32
unsigned __stdcall
#else
void*
#endif
worker_fn(void* arg) {
    Worker* w = (Worker*)arg;
    int last_phase = 0;

    while (!ATOMIC_LOAD(&g_shutdown)) {
        /* Wait for new work (spin briefly, then sleep) */
        int phase = ATOMIC_LOAD(&g_task.phase);
        if (phase == last_phase) {
            /* Spin for ~1µs (check ~200 times) */
            int spins = 200;
            while (spins-- > 0) {
                phase = ATOMIC_LOAD(&g_task.phase);
                if (phase != last_phase) break;
                tp_cpu_relax();
            }
            if (phase == last_phase) {
                /* Sleep until woken */
                #ifdef _WIN32
                WaitOnAddress((volatile void*)&g_task.phase, &last_phase, sizeof(int), INFINITE);
                #else
                syscall(SYS_futex, &g_task.phase, FUTEX_WAIT, last_phase, NULL, NULL, 0);
                #endif
                continue;
            }
        }
        last_phase = phase;

        /* Execute work chunks */
        while (1) {
            int idx = ATOMIC_ADD(&g_task.next_idx, g_task.grain);
            if (idx >= g_task.total) break;
            int end = idx + g_task.grain;
            if (end > g_task.total) end = g_task.total;
            g_task.fn(g_task.ctx, idx, end);
        }

        /* Signal done */
        ATOMIC_ADD(&g_task.done_count, 1);
    }

    #ifdef _WIN32
    return 0;
    #else
    return NULL;
    #endif
}

void tp_init(int n_threads) {
    if (n_threads <= 0) {
        #ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        n_threads = si.dwNumberOfProcessors;
        #else
        n_threads = sysconf(_SC_NPROCESSORS_ONLN);
        #endif
        if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
        if (n_threads < 1) n_threads = 1;
    }
    /* Use n_threads-1 workers (main thread also participates) */
    g_n_threads = n_threads;
    int n_workers = n_threads - 1;
    if (n_workers > MAX_THREADS) n_workers = MAX_THREADS;

    memset(&g_task, 0, sizeof(g_task));
    ATOMIC_STORE(&g_shutdown, 0);

    for (int i = 0; i < n_workers; i++) {
        g_workers[i].id = i;
        g_workers[i].alive = 1;
        #ifdef _WIN32
        g_workers[i].handle = (HANDLE)_beginthreadex(NULL, 0, worker_fn, &g_workers[i], 0, NULL);
        #else
        pthread_create(&g_workers[i].handle, NULL, worker_fn, &g_workers[i]);
        #endif
    }
}

void tp_parallel_for(tp_task_fn fn, void* ctx, int total, int grain) {
    if (total <= 0) return;
    if (grain <= 0) grain = 1;

    int n_workers = g_n_threads - 1;
    if (n_workers <= 0 || total <= grain) {
        /* Single-threaded fast path */
        fn(ctx, 0, total);
        return;
    }

    /* Set up task */
    g_task.fn = fn;
    g_task.ctx = ctx;
    g_task.total = total;
    g_task.grain = grain;
    ATOMIC_STORE(&g_task.next_idx, 0);
    ATOMIC_STORE(&g_task.done_count, 0);
    g_task.n_workers = n_workers;

    /* Wake workers */
    ATOMIC_ADD(&g_task.phase, 1);
    #ifdef _WIN32
    WakeByAddressAll((void*)&g_task.phase);
    #else
    syscall(SYS_futex, &g_task.phase, FUTEX_WAKE, n_workers, NULL, NULL, 0);
    #endif

    /* Main thread participates too */
    while (1) {
        int idx = ATOMIC_ADD(&g_task.next_idx, grain);
        if (idx >= total) break;
        int end = idx + grain;
        if (end > total) end = total;
        fn(ctx, idx, end);
    }

    /* Wait for workers to finish */
    while (ATOMIC_LOAD(&g_task.done_count) < n_workers) {
        tp_cpu_relax();
    }
}

void tp_destroy(void) {
    ATOMIC_STORE(&g_shutdown, 1);
    ATOMIC_ADD(&g_task.phase, 1);
    #ifdef _WIN32
    WakeByAddressAll((void*)&g_task.phase);
    for (int i = 0; i < g_n_threads - 1; i++)
        WaitForSingleObject(g_workers[i].handle, INFINITE);
    #else
    syscall(SYS_futex, &g_task.phase, FUTEX_WAKE, MAX_THREADS, NULL, NULL, 0);
    for (int i = 0; i < g_n_threads - 1; i++)
        pthread_join(g_workers[i].handle, NULL);
    #endif
}

int tp_num_threads(void) { return g_n_threads; }
