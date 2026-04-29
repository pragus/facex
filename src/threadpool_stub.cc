/*
 * threadpool_stub.c — Single-threaded stub for WASM build.
 * All work runs sequentially on the main thread.
 */

typedef void (*tp_task_fn)(void* ctx, int start, int end);

void tp_init(int n_threads) {
    (void)n_threads;
}

void tp_parallel_for(tp_task_fn fn, void* ctx, int total, int grain) {
    if (grain <= 0) grain = 1;
    for (int i = 0; i < total; i += grain) {
        int end = i + grain;
        if (end > total) end = total;
        fn(ctx, i, end);
    }
}

void tp_shutdown(void) {}
