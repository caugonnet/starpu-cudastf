#include <starpu.h>
#include <stdio.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

starpu_data_handle_t _handle_y, _handle_x;

#ifndef STARPU_USE_CUDA
#error CUDA support needed in StarPU
#endif

__global__ void scale_kernel(double alpha, double *x, unsigned n)
{
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] *= alpha;
    }
}

void scale_gpu(void *descr[], void *arg) {
    cudaStream_t stream = starpu_cuda_get_local_stream();
    unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
    double *out = (double *)STARPU_VECTOR_GET_PTR(descr[0]);

    // scale by 1.1
    scale_kernel<<<16, 32, 0, stream>>>(1.1, out, n);
//    cudaStreamSynchronize(stream);
}

struct starpu_codelet scale_cl {
    .cuda_funcs = {scale_gpu},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 1,
    .modes = {STARPU_RW}
};

__global__ void add_kernel(double *out, double *in, unsigned n)
{
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] += in[i];
    }
}

void add_gpu(void *descr[], void *arg) {
    cudaStream_t stream = starpu_cuda_get_local_stream();
    unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
    double *out = (double *)STARPU_VECTOR_GET_PTR(descr[0]);
    double *in  = (double *)STARPU_VECTOR_GET_PTR(descr[1]);

    // out += in;
    add_kernel<<<16, 32, 0, stream>>>(out, in, n);
}

struct starpu_codelet add_cl {
    .cuda_funcs = {add_gpu},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R}
};

__global__ void add2_kernel(double *out, double *in1, double *in2, unsigned n)
{
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = in1[i] + in2[i];
    }
}

void add2_gpu(void *descr[], void *arg) {
    cudaStream_t stream = starpu_cuda_get_local_stream();
    unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
    double *out = (double *)STARPU_VECTOR_GET_PTR(descr[0]);
    double *in1 = (double *)STARPU_VECTOR_GET_PTR(descr[1]);
    double *in2 = (double *)STARPU_VECTOR_GET_PTR(descr[2]);

    // out = in1 + in2;
    add2_kernel<<<16, 32, 0, stream>>>(out, in1, in2, n);
//    cudaStreamSynchronize(stream);
}

struct starpu_codelet add2_cl {
    .cuda_funcs = {add2_gpu},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_RW, STARPU_R, STARPU_R}
};



void algo_gpu(void *descr[], void *arg)
{
    cudaStream_t stream = starpu_cuda_get_local_stream();

    unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

    double *x = (double *)STARPU_VECTOR_GET_PTR(descr[0]);
    double *y = (double *)STARPU_VECTOR_GET_PTR(descr[1]);
    double *z = (double *)STARPU_VECTOR_GET_PTR(descr[2]);

    // z *= 1.1
    scale_kernel<<<16, 32, 0, stream>>>(1.1, z, n);
    // x += z;
    add_kernel<<<16, 32, 0, stream>>>(x, z, n);
    // y += z;
    add_kernel<<<16, 32, 0, stream>>>(y, z, n);
    // z = x + y;
    add2_kernel<<<16, 32, 0, stream>>>(z, x, y, n);

//    cudaStreamSynchronize(stream);
}

struct starpu_codelet algo_cl {
    .cuda_funcs = {algo_gpu},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_RW, STARPU_RW, STARPU_RW}
};



void algo_cudastf_gpu(void *descr[], void *arg)
{
    cudaStream_t stream = starpu_cuda_get_local_stream();

    auto *async_handle = ((async_resources_handle *)arg);

    unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

    double *x = (double *)STARPU_VECTOR_GET_PTR(descr[0]);
    double *y = (double *)STARPU_VECTOR_GET_PTR(descr[1]);
    double *z = (double *)STARPU_VECTOR_GET_PTR(descr[2]);


//    context ctx = stream_ctx(stream, *async_handle);
    context ctx = graph_ctx(stream, *async_handle);

    auto lX = ctx.logical_data(make_slice(x, n), data_place::current_device());
    auto lY = ctx.logical_data(make_slice(y, n), data_place::current_device());
    auto lZ = ctx.logical_data(make_slice(z, n), data_place::current_device());

    // z *= 1.1
    ctx.task(lZ.rw())->*[n](cudaStream_t s, auto dz) {
        scale_kernel<<<16, 32, 0, s>>>(1.1, dz.data_handle(), n);
    };

    // x += z;
    ctx.task(lX.rw(), lZ.read())->*[n](cudaStream_t s, auto dx, auto dz) {
        add_kernel<<<16, 32, 0, s>>>(dx.data_handle(), dz.data_handle(), n);
    };

    // y += z;
    ctx.task(lY.rw(), lZ.read())->*[n](cudaStream_t s, auto dy, auto dz) {
        add_kernel<<<16, 32, 0, s>>>(dy.data_handle(), dz.data_handle(), n);
    };

    // z = x + y;
    ctx.task(lX.read(), lY.read(), lZ.write())->*[n](cudaStream_t s, auto dx, auto dy, auto dz) {
        add2_kernel<<<16, 32, 0, s>>>(dz.data_handle(), dx.data_handle(), dy.data_handle(), n);
    };

    ctx.finalize();

//    cudaStreamSynchronize(stream);
}

     struct starpu_codelet algo_cudastf_cl {
         .cuda_funcs = {algo_cudastf_gpu},
         .cuda_flags = {STARPU_CUDA_ASYNC},
         .nbuffers = 3,
         .modes = {STARPU_RW, STARPU_RW, STARPU_RW}
     };



void cpu_func(void *buffers[], void *cl_arg)
{
    printf("Hello world\n");
}
 
struct starpu_codelet cl =
{
    .cpu_funcs = { cpu_func },
    .nbuffers = 0
};

int main(int argc, char *argv[])
{
    const size_t N = 1024*1024;

    int ret = starpu_init(NULL);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    double *X, *Y, *Z;
    starpu_malloc((void **)&X, N*sizeof(double));
    starpu_malloc((void **)&Y, N*sizeof(double));
    starpu_malloc((void **)&Z, N*sizeof(double));

    /* Declare the data to StarPU */
    starpu_data_handle_t handle_x, handle_y, handle_z;
    starpu_vector_data_register(&handle_x, STARPU_MAIN_RAM, (uintptr_t)X, N, sizeof(double));
    starpu_vector_data_register(&handle_y, STARPU_MAIN_RAM, (uintptr_t)Y, N, sizeof(double));
    starpu_vector_data_register(&handle_z, STARPU_MAIN_RAM, (uintptr_t)Z, N, sizeof(double));

    double alpha = 3.14;

    nvtxRangePushA("warmup");
    for (size_t iter = 0; iter < 10; iter++) {
        struct starpu_task *t;

        t = starpu_task_create();
        t->cl = &scale_cl;
        t->handles[0] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add_cl;
        t->handles[0] = handle_x;
        t->handles[1] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add_cl;
        t->handles[0] = handle_y;
        t->handles[1] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add2_cl;
        t->handles[0] = handle_z;
        t->handles[1] = handle_x;
        t->handles[2] = handle_y;
        starpu_task_submit(t);
    }
    starpu_task_wait_for_all();
    nvtxRangePop();




    nvtxRangePushA("fine-starpu");
    for (size_t iter = 0; iter < 10; iter++) {
        struct starpu_task *t;

        t = starpu_task_create();
        t->cl = &scale_cl;
        t->handles[0] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add_cl;
        t->handles[0] = handle_x;
        t->handles[1] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add_cl;
        t->handles[0] = handle_y;
        t->handles[1] = handle_z;
        starpu_task_submit(t);

        t = starpu_task_create();
        t->cl = &add2_cl;
        t->handles[0] = handle_z;
        t->handles[1] = handle_x;
        t->handles[2] = handle_y;
        starpu_task_submit(t);
    }
    starpu_task_wait_for_all();
    nvtxRangePop();


    nvtxRangePushA("starpu");
    for (size_t iter = 0; iter < 10; iter++) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &algo_cl;
        task->handles[0] = handle_x;
        task->handles[1] = handle_y;
        task->handles[2] = handle_z;

        ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }
    starpu_task_wait_for_all();
    nvtxRangePop();

    nvtxRangePushA("starpu-cudastf");
    async_resources_handle async_handle;

    for (size_t iter = 0; iter < 10; iter++) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &algo_cudastf_cl;
        task->cl_arg = &async_handle;
        task->cl_arg_size = sizeof(async_handle);
        task->handles[0] = handle_x;
        task->handles[1] = handle_y;
        task->handles[2] = handle_z;

        ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }
    starpu_task_wait_for_all();
    nvtxRangePop();

    nvtxRangePushA("only-cudastf");
    {
    context ctx = stream_ctx();

    auto lX = ctx.logical_data(make_slice(X, N));
    auto lY = ctx.logical_data(make_slice(Y, N));
    auto lZ = ctx.logical_data(make_slice(Z, N));
    lX.set_write_back(false);
    lY.set_write_back(false);
    lZ.set_write_back(false);

    // Load data on the device once to obtain the same situation as StarPU and
    // ignore data transfers
    ctx.task(lX.read(), lY.read(), lZ.read()).set_symbol("data preload")->*[](cudaStream_t, auto, auto, auto) {};
    cudaStreamSynchronize(ctx.task_fence());

    nvtxRangePushA("only-cudastf-compute");
    for (size_t iter = 0; iter < 10; iter++) {
        // z *= 1.1
        ctx.task(lZ.rw())->*[N](cudaStream_t s, auto dz) {
            scale_kernel<<<16, 32, 0, s>>>(1.1, dz.data_handle(), N);
        };

        // x += z;
        ctx.task(lX.rw(), lZ.read())->*[N](cudaStream_t s, auto dx, auto dz) {
            add_kernel<<<16, 32, 0, s>>>(dx.data_handle(), dz.data_handle(), N);
        };

        // y += z;
        ctx.task(lY.rw(), lZ.read())->*[N](cudaStream_t s, auto dy, auto dz) {
            add_kernel<<<16, 32, 0, s>>>(dy.data_handle(), dz.data_handle(), N);
        };

        // z = x + y;
        ctx.task(lX.read(), lY.read(), lZ.write())->*[N](cudaStream_t s, auto dx, auto dy, auto dz) {
            add2_kernel<<<16, 32, 0, s>>>(dz.data_handle(), dx.data_handle(), dy.data_handle(), N);
        };
    }
    ctx.finalize();
    nvtxRangePop();
    }
    nvtxRangePop();



    starpu_shutdown();

    return 0; 
}
