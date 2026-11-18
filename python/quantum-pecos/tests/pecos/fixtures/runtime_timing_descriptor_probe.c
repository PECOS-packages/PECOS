/* Force a reader into descriptor initialization without timing-based sleeps. */

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static bool published, release_writer, writer_done, reader_at_once,
    reader_read_init;
static _Thread_local bool is_reader;
static void publication_pause(void) {
    pthread_mutex_lock(&gate);
    published = true;
    pthread_cond_broadcast(&changed);
    while (!release_writer)
        pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
}
static int observed_once(pthread_once_t *control, void (*fn)(void)) {
    if (is_reader) {
        pthread_mutex_lock(&gate);
        reader_at_once = true;
        pthread_cond_broadcast(&changed);
        pthread_mutex_unlock(&gate);
    }
    return pthread_once(control, fn);
}
#define pthread_once observed_once

#include "timing_proxy_instrumented.c"

#undef pthread_once
static int observed_errno, base_init_seen, wrapped_start_seen;
static uint32_t runtime_argc;
static const char *const *runtime_argv;
static void *writer(void *unused) {
    (void)unused;
    assert(selene_runtime_get_plugin_descriptor_v1());
    pthread_mutex_lock(&gate);
    writer_done = true;
    pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    return NULL;
}
static void *reader(void *unused) {
    (void)unused;
    is_reader = true;
    const SeleneRuntimePluginDescriptorV1 *d =
        selene_runtime_get_plugin_descriptor_v1();
    assert(d);
    /* Model separate loads in the host's non-atomic descriptor copy. */
    __typeof__(d->init_fn) copied_init = d->init_fn;
    pthread_mutex_lock(&gate);
    reader_read_init = true;
    pthread_cond_broadcast(&changed);
    while (!writer_done)
        pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    __typeof__(d->shot_start_fn) copied_start = d->shot_start_fn;
    base_init_seen = copied_init == original.init_fn;
    wrapped_start_seen = copied_start == start;
    RuntimeInstance h = NULL;
    assert(copied_init(&h, 1, 0, runtime_argc, runtime_argv) == 0);
    observed_errno = copied_start(h, 0, 42);
    assert(exit_runtime(h) == 0);
    return NULL;
}
int main(int argc, char **argv) {
    runtime_argc = (uint32_t)(argc - 1);
    runtime_argv = (const char *const *)(argv + 1);
    pthread_t w, r;
    assert(!pthread_create(&w, NULL, writer, NULL));
    pthread_mutex_lock(&gate);
    while (!published)
        pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    assert(!pthread_create(&r, NULL, reader, NULL));
    pthread_mutex_lock(&gate);
    /* Old getter returns immediately; fixed getter enters real pthread_once.
     * Either event proves the reader reached the getter before writer resumes.
     */
    while (!reader_at_once && !reader_read_init)
        pthread_cond_wait(&changed, &gate);
    release_writer = true;
    pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&gate);
    assert(!pthread_join(w, NULL));
    assert(!pthread_join(r, NULL));
    printf("base_init=%d wrapped_start=%d errno=%d\n", base_init_seen,
           wrapped_start_seen, observed_errno);
    return observed_errno == 0 ? 0 : 1;
}
