/*
 * threadpool_wasm.c — Thread pool for WASM (Emscripten pthreads).
 * Uses pthread_cond for signaling instead of futex/WaitOnAddress.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

typedef void (*tp_task_fn)(void* ctx, int start, int end);

#define MAX_WORKERS 4

static struct {
    tp_task_fn fn;
    void* ctx;
    int total;
    int grain;
    atomic_int next_idx;
    atomic_int done_count;
    int phase;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
} g_task;

static struct {
    pthread_t handle;
    int id;
} g_workers[MAX_WORKERS];

static int n_workers = 0;
static int g_shutdown = 0;

static void* worker_fn(void* arg) {
    (void)arg;
    int last_phase = 0;

    while (!g_shutdown) {
        /* Wait for work */
        pthread_mutex_lock(&g_task.mutex);
        while (g_task.phase == last_phase && !g_shutdown)
            pthread_cond_wait(&g_task.cond_work, &g_task.mutex);
        pthread_mutex_unlock(&g_task.mutex);

        if (g_shutdown) break;
        last_phase = g_task.phase;

        /* Execute chunks */
        while (1) {
            int idx = atomic_fetch_add(&g_task.next_idx, g_task.grain);
            if (idx >= g_task.total) break;
            int end = idx + g_task.grain;
            if (end > g_task.total) end = g_task.total;
            g_task.fn(g_task.ctx, idx, end);
        }

        /* Signal done */
        int prev = atomic_fetch_add(&g_task.done_count, 1);
        if (prev + 1 >= n_workers) {
            pthread_mutex_lock(&g_task.mutex);
            pthread_cond_signal(&g_task.cond_done);
            pthread_mutex_unlock(&g_task.mutex);
        }
    }
    return NULL;
}

void tp_init(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_WORKERS + 1) n_threads = MAX_WORKERS + 1;
    n_workers = n_threads - 1; /* main thread also participates */

    pthread_mutex_init(&g_task.mutex, NULL);
    pthread_cond_init(&g_task.cond_work, NULL);
    pthread_cond_init(&g_task.cond_done, NULL);
    g_task.phase = 0;

    for (int i = 0; i < n_workers; i++) {
        g_workers[i].id = i;
        pthread_create(&g_workers[i].handle, NULL, worker_fn, &g_workers[i]);
    }
}

void tp_parallel_for(tp_task_fn fn, void* ctx, int total, int grain) {
    if (grain <= 0) grain = 1;

    if (n_workers <= 0 || total <= grain) {
        /* Single-threaded fallback */
        for (int i = 0; i < total; i += grain) {
            int end = i + grain;
            if (end > total) end = total;
            fn(ctx, i, end);
        }
        return;
    }

    /* Setup task */
    g_task.fn = fn;
    g_task.ctx = ctx;
    g_task.total = total;
    g_task.grain = grain;
    atomic_store(&g_task.next_idx, 0);
    atomic_store(&g_task.done_count, 0);

    /* Wake workers */
    pthread_mutex_lock(&g_task.mutex);
    g_task.phase++;
    pthread_cond_broadcast(&g_task.cond_work);
    pthread_mutex_unlock(&g_task.mutex);

    /* Main thread participates */
    while (1) {
        int idx = atomic_fetch_add(&g_task.next_idx, grain);
        if (idx >= total) break;
        int end = idx + grain;
        if (end > total) end = total;
        fn(ctx, idx, end);
    }

    /* Wait for workers */
    pthread_mutex_lock(&g_task.mutex);
    while (atomic_load(&g_task.done_count) < n_workers)
        pthread_cond_wait(&g_task.cond_done, &g_task.mutex);
    pthread_mutex_unlock(&g_task.mutex);
}

void tp_shutdown(void) {
    g_shutdown = 1;
    pthread_mutex_lock(&g_task.mutex);
    g_task.phase++;
    pthread_cond_broadcast(&g_task.cond_work);
    pthread_mutex_unlock(&g_task.mutex);
    for (int i = 0; i < n_workers; i++)
        pthread_join(g_workers[i].handle, NULL);
}
