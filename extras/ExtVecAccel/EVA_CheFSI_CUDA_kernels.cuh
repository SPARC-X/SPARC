// TODO: Store [n{x, y, z, xyz, x_ex, y_ex, z_ex, xyz_ex}, radius] in a constant array?

// ============== Auxiliary kernels for the Laplacian kernel ===============

__global__ void Lap_MV_orth_copy_x_kernel(
    const int nx, const int ny, const int nz, const int radius, 
    const double *x, const int ncol, double *x_ex
)
{
    const int nxyz    = nx * ny * nz;
    const int nx_ex   = nx + 2 * radius;
    const int ny_ex   = ny + 2 * radius;
    const int nz_ex   = nz + 2 * radius;
    const int nxyz_ex = nx_ex * ny_ex * nz_ex;
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int ip = i + radius;
    const int jp = j + radius;
    const int kp = k + radius;
    
    bool valid_x = true;
    if (i >= nx || j >= ny || k >= nz) valid_x = false;
    
    for (int n = 0; n < ncol; n++)
    {
        const int idx    = n * nxyz    + k  * nx    * ny    + j  * nx    + i;
        const int idx_ex = n * nxyz_ex + kp * nx_ex * ny_ex + jp * nx_ex + ip;
        if (valid_x == true) x_ex[idx_ex] = x[idx];
        __syncthreads();  // Comment this line?
    }
}

__global__ void Lap_MV_orth_period_BC_kernel(
    const int nx, const int ny, const int nz, const int radius, 
    const int i_spos, const int i_len, const int ip_spos,
    const int j_spos, const int j_len, const int jp_spos,
    const int k_spos, const int k_len, const int kp_spos,
    const double *x,  const int ncol,  double *x_ex
)
{
    const int nxyz    = nx * ny * nz;
    const int nx_ex   = nx + 2 * radius;
    const int ny_ex   = ny + 2 * radius;
    const int nz_ex   = nz + 2 * radius;
    const int nxyz_ex = nx_ex * ny_ex * nz_ex;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int ip = ip_spos + i;
    const int jp = jp_spos + j;
    const int kp = kp_spos + k;
    const int ii = i + i_spos;
    const int jj = j + j_spos;
    const int kk = k + k_spos;
    
    bool valid_x = true;
    if (i >= i_len || j >= j_len || k >= k_len) valid_x = false;
    
    for (int n = 0; n < ncol; n++)
    {
        const int idx    = n * nxyz    + kk * nx    * ny    + jj * nx    + ii;
        const int idx_ex = n * nxyz_ex + kp * nx_ex * ny_ex + jp * nx_ex + ip;
        if (valid_x == true) x_ex[idx_ex] = x[idx];
       __syncthreads();  // Comment this line?
    }
}

__global__ void Lap_MV_orth_zero_BC_kernel(
    const int nx, const int ny, const int nz, const int radius, 
    const int ip_len, const int ip_spos,
    const int jp_len, const int jp_spos,
    const int kp_len, const int kp_spos,
    const double *x,  const int ncol, double *x_ex
)
{
    const int nx_ex   = nx + 2 * radius;
    const int ny_ex   = ny + 2 * radius;
    const int nz_ex   = nz + 2 * radius;
    const int nxyz_ex = nx_ex * ny_ex * nz_ex;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const int ip = ip_spos + i;
    const int jp = jp_spos + j;
    const int kp = kp_spos + k;
    
    bool valid_x = true;
    if (i >= ip_len || j >= jp_len || k >= kp_len) valid_x = false;
    
    for (int n = 0; n < ncol; n++)
    {
        const int idx_ex = n * nxyz_ex + kp * nx_ex * ny_ex + jp * nx_ex + ip;
        if (valid_x == true) x_ex[idx_ex] = 0.0;
        __syncthreads();  // Comment this line?
    }
}    

// ==================== Laplacian 3-axis stencil kernel ====================

// Note: If you change the RADIUS, you should also change the unrolling below
#define RADIUS 6

#define X_BLK_SIZE 32
#define Y_BLK_SIZE 16
#define X_BLK_2R   (X_BLK_SIZE + 2 * (RADIUS))
#define Y_BLK_2R   (Y_BLK_SIZE + 2 * (RADIUS))

// Orthogonal Laplacian stencil coefficients
__constant__ double cu_Lap_x_orth[RADIUS + 1];  
__constant__ double cu_Lap_y_orth[RADIUS + 1];
__constant__ double cu_Lap_z_orth[RADIUS + 1];

// Kernel for calculating (a * Lap + b * diag(v) + c * I) * x
__global__ void Lap_orth_r6_kernel(
    const int nx, const int ny, const int nz, 
    const double coef_0, const double b, 
    const double *_x0, const double *v, double *_x1
)
{
    const int local_x  = threadIdx.x;
    const int local_y  = threadIdx.y;
    const int tile_x   = local_x + RADIUS;
    const int tile_y   = local_y + RADIUS;
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_z = nx * ny;
    const int stride_y_ex = nx + 2 * RADIUS;
    const int stride_z_ex = stride_y_ex * (ny + 2 * RADIUS);

    const int icol = blockIdx.z;
    const int nxyz = stride_z * nz;
    const int nxyz_ex = stride_z_ex * (nz + 2 * RADIUS);
    const double *x0 = _x0 + icol * nxyz_ex;
    double *x1 = _x1 + icol * nxyz;
    
    int thread_x0_idx, thread_x0_ridx, thread_x1_idx;
    thread_x0_idx  = RADIUS * stride_z_ex;
    thread_x0_idx += (global_y + RADIUS) * stride_y_ex;
    thread_x0_idx += (global_x + RADIUS);
    thread_x1_idx  = global_y * nx + global_x;
    thread_x0_ridx = thread_x0_idx - RADIUS * stride_z_ex;
    
    bool valid_x1 = true;
    if ((global_x >= nx) || (global_y >= ny)) valid_x1 = false;
    
    double z_axis_buff[2 * RADIUS + 1];
    __shared__ double xy_plane[Y_BLK_2R][X_BLK_2R];
    
    // Prefetch z-axis front and behind data
    for (int iz = -RADIUS; iz < RADIUS; iz++)
    {
        // +1 here because we will advance the z index first
        z_axis_buff[iz + RADIUS + 1] = x0[thread_x0_ridx];
        thread_x0_ridx += stride_z_ex;
    }
    
    // Step through the xy-planes
    for (int iz = 0; iz < nz; iz++)
    {   
        // 1. Advance the z-axis thread buffer
        #pragma unroll 12
        for (int i = 0; i < 2 * RADIUS; i++)
            z_axis_buff[i] = z_axis_buff[i + 1];
        if (valid_x1) z_axis_buff[2 * RADIUS] = x0[thread_x0_ridx];
        thread_x0_ridx += stride_z_ex;
        
        //__syncthreads();
        
        // 2. Load the x & y halo for current z
        if (local_y < RADIUS)
        {
            xy_plane[local_y][tile_x]             = x0[thread_x0_idx -     RADIUS * stride_y_ex];
            xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[thread_x0_idx + Y_BLK_SIZE * stride_y_ex];
        }
        
        if (local_x < RADIUS)
        {
            xy_plane[tile_y][local_x]             = x0[thread_x0_idx - RADIUS];
            xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[thread_x0_idx + X_BLK_SIZE];
        }

        //double current_x0_z = z_axis_buff[RADIUS];  // Error?
        double current_x0_z = x0[thread_x0_idx];
        xy_plane[tile_y][tile_x] = current_x0_z;
        __syncthreads();
        
        // 3. Stencil calculation
        double value = coef_0 * current_x0_z;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            value += cu_Lap_z_orth[r] * (z_axis_buff[RADIUS + r]      + z_axis_buff[RADIUS - r]     );
            value += cu_Lap_y_orth[r] * (xy_plane[tile_y + r][tile_x] + xy_plane[tile_y - r][tile_x]);
            value += cu_Lap_x_orth[r] * (xy_plane[tile_y][tile_x + r] + xy_plane[tile_y][tile_x - r]);
        }
        
        // 4. Store the output value
        if (valid_x1) x1[thread_x1_idx] = value + b * (current_x0_z * v[thread_x1_idx]);
        
        thread_x1_idx += stride_z;
        thread_x0_idx += stride_z_ex;
        __syncthreads();
    }
}

// Nonorthogonal Laplacian / gradient stencil coefficients
__constant__ double cu_Lap_wt_nonorth [(RADIUS + 1) * 5];
__constant__ double cu_Grad_wt_nonorth[(RADIUS + 1) * 8];

__global__ void DY_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const int x0_stride_x, const int x0_stride_y, const int x1_stride_y, 
    const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int DMnd_Dx,     const int DMnd_ex,     const int Grad_stencil_offset,
    const double coef_0,   const double *_x0,     double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + ix_offset;
    const int x0_y = x1_y + iy_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    double *x1 = _x1 + icol * DMnd_Dx;
    
    __shared__ double xy_plane[Y_BLK_2R][X_BLK_SIZE];
    
    bool valid_x1 = true, valid_x0_load = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx) || (x1_y >= ny + RADIUS)) valid_x0_load = false;
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + iz_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        
        if (valid_x0_load) xy_plane[tile_y][local_x] = x0[idx_x0];
        else xy_plane[tile_y][local_x] = 0.0;
        if (local_y < RADIUS)
        {
            xy_plane[local_y][local_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            xy_plane[tile_y + Y_BLK_SIZE][local_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
        }
        __syncthreads();

        double res = xy_plane[tile_y][local_x] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            double coef = cu_Grad_wt_nonorth[Grad_stencil_offset + r];
            res += (xy_plane[tile_y + r][local_x] - xy_plane[tile_y - r][local_x]) * coef;
        }

        if (valid_x1) x1[idx_x1] = res;
        __syncthreads();
    }
}

__global__ void DZ_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const int x0_stride_x, const int x0_stride_y, const int x1_stride_y, 
    const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int DMnd_Dx,     const int DMnd_ex,     const int Grad_stencil_offset,
    const double coef_0,   const double *_x0,     double *_x1
)
{
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + ix_offset;
    const int x0_y = x1_y + iy_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    double *x1 = _x1 + icol * DMnd_Dx;
    
    double z_buf[2 * RADIUS + 1];

    bool valid_x1 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;

    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + iz_offset * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }

    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + iz_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 2 * RADIUS; i++) z_buf[i] = z_buf[i + 1];
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        
        double res = z_buf[RADIUS] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            double coef = cu_Grad_wt_nonorth[Grad_stencil_offset + r];
            res += (z_buf[RADIUS + r] - z_buf[RADIUS - r]) * coef;
        }

        if (valid_x1) x1[idx_x1] = res;
    }
}

__global__ void DX_DY_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const int x0_stride_0, const int x0_stride_1, const int x0_stride_y, 
    const int x1_stride_y, const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int DMnd_Dx,     const int DMnd_ex,     
    const int Grad_stencil_offset0, const int Grad_stencil_offset1,
    const double *_x0,     double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + ix_offset;
    const int x0_y = x1_y + iy_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    double *x1 = _x1 + icol * DMnd_Dx;
    
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_2R];
    
    bool valid_x1 = true, valid_x0 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx + RADIUS) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + iz_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        
        if (valid_x0) x0_xy_plane[tile_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][tile_x] = 0.0;
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][tile_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
        }
        if (local_x < RADIUS)
        {
            x0_xy_plane[tile_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
        }
        __syncthreads();
        
        double res0 = 0.0, res1 = 0.0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            double coef0 = cu_Grad_wt_nonorth[Grad_stencil_offset0 + r];
            double coef1 = cu_Grad_wt_nonorth[Grad_stencil_offset1 + r];
            res0 += (x0_xy_plane[tile_y][tile_x + r] - x0_xy_plane[tile_y][tile_x - r]) * coef0;
            res1 += (x0_xy_plane[tile_y + r][tile_x] - x0_xy_plane[tile_y - r][tile_x]) * coef1;
        }
        
        if (valid_x1) x1[idx_x1] = res0 + res1;
        __syncthreads();
    }
}

__global__ void DX_DZ_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const int x0_stride_0, const int x0_stride_1, const int x0_stride_y, 
    const int x1_stride_y, const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int DMnd_Dx,     const int DMnd_ex,     
    const int Grad_stencil_offset0, const int Grad_stencil_offset1,
    const double *_x0,     double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + ix_offset;
    const int x0_y = x1_y + iy_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    double *x1 = _x1 + icol * DMnd_Dx;
    
    double z_buf[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_SIZE][X_BLK_2R];
    
    bool valid_x1 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + iz_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 12; i++) z_buf[i] = z_buf[i + 1];
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        
        if (x1_x < nx + RADIUS) x0_xy_plane[local_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[local_y][tile_x] = 0.0;
        if (local_x < RADIUS)
        {
            x0_xy_plane[local_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[local_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
        }
        __syncthreads();
        
        double res0 = 0.0, res1 = 0.0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            double coef0 = cu_Grad_wt_nonorth[Grad_stencil_offset0 + r];
            double coef1 = cu_Grad_wt_nonorth[Grad_stencil_offset1 + r];
            res0 += (x0_xy_plane[local_y][tile_x + r] - x0_xy_plane[local_y][tile_x - r]) * coef0;
            res1 += (z_buf[RADIUS + r] - z_buf[RADIUS - r]) * coef1;
        }
        
        if (valid_x1) x1[idx_x1] = res0 + res1;
        __syncthreads();
    }
}

__global__ void DY_DZ_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const int x0_stride_0, const int x0_stride_1, const int x0_stride_y, 
    const int x1_stride_y, const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int DMnd_Dx,     const int DMnd_ex,     
    const int Grad_stencil_offset0, const int Grad_stencil_offset1,
    const double *_x0,     double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + ix_offset;
    const int x0_y = x1_y + iy_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    double *x1 = _x1 + icol * DMnd_Dx;
    
    double z_buf[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_SIZE];
    
    bool valid_x1 = true, valid_x0 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + iz_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 12; i++) z_buf[i] = z_buf[i + 1];
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        
        if (valid_x0) x0_xy_plane[tile_y][local_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][local_x] = 0.0;
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][local_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][local_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
        }
        __syncthreads();
        
        double res0 = 0.0, res1 = 0.0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            double coef0 = cu_Grad_wt_nonorth[Grad_stencil_offset0 + r];
            double coef1 = cu_Grad_wt_nonorth[Grad_stencil_offset1 + r];
            res0 += (x0_xy_plane[tile_y + r][local_x] - x0_xy_plane[tile_y - r][local_x]) * coef0;
            res1 += (z_buf[RADIUS + r] - z_buf[RADIUS - r]) * coef1;
        }
        
        if (valid_x1) x1[idx_x1] = res0 + res1;
        __syncthreads();
    }
}

__global__ void Lap_nonorth_DX_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const double *_x0,     const double *_Dx,     const int stride_Dx, 
    const int DMnd,        const int DMnd_ex,     const int DMnd_Dx, 
    const int x1_stride_y, const int x0_stride_y, const int Dx_stride_y, 
    const int x1_stride_z, const int x0_stride_z, const int Dx_stride_z,
    const int x0_x_offset, const int x0_y_offset, const int x0_z_offset,
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset,
    const double coef_0, const double b, const double *v0, double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + x0_x_offset;
    const int x0_y = x1_y + x0_y_offset;
    const int Dx_x = x1_x + Dx_x_offset;
    const int Dx_y = x1_y + Dx_y_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int idx_Dx_xy = Dx_y * Dx_stride_y + Dx_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    const double *Dx = _Dx + icol * DMnd_Dx;
    double *x1 = _x1 + icol * DMnd;
    
    double z_buf[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_2R];
    __shared__ double Dx_xy_plane[Y_BLK_SIZE][X_BLK_2R];
    
    bool valid_x1 = true, valid_x0 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx + RADIUS) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + x0_z_offset;
        int Dx_z = x1_z + Dx_z_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        int idx_Dx = idx_Dx_xy + Dx_z * Dx_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 12; i++) z_buf[i] = z_buf[i + 1];
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        
        if (valid_x0) x0_xy_plane[tile_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][tile_x] = 0.0;
        
        if (x1_x < nx + RADIUS) Dx_xy_plane[local_y][tile_x] = Dx[idx_Dx];
        else Dx_xy_plane[local_y][tile_x] = 0.0;
        
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][tile_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
        }
        if (local_x < RADIUS)
        {
            x0_xy_plane[tile_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
            Dx_xy_plane[local_y][local_x]             = Dx[idx_Dx - RADIUS];
            Dx_xy_plane[local_y][tile_x + X_BLK_SIZE] = Dx[idx_Dx + X_BLK_SIZE];
        }
        __syncthreads(); 

        double res = z_buf[RADIUS] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            int r_fac = 4 * r + 1;
            res += (x0_xy_plane[tile_y][tile_x - r] + x0_xy_plane[tile_y][tile_x + r]) * cu_Lap_wt_nonorth[r_fac];
            res += (x0_xy_plane[tile_y - r][tile_x] + x0_xy_plane[tile_y + r][tile_x]) * cu_Lap_wt_nonorth[r_fac+1];
            res += (z_buf[RADIUS - r] + z_buf[RADIUS + r]) * cu_Lap_wt_nonorth[r_fac+2];
            res += (Dx_xy_plane[local_y][tile_x + r] - Dx_xy_plane[local_y][tile_x - r]) * cu_Lap_wt_nonorth[r_fac+3];
        }
            
        if (valid_x1) x1[idx_x1] = res + b * (v0[idx_x1] * x0[idx_x0]);
        __syncthreads(); 
    }
}

__global__ void Lap_nonorth_DY_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const double *_x0,     const double *_Dx,     const int stride_Dx, 
    const int DMnd,        const int DMnd_ex,     const int DMnd_Dx, 
    const int x1_stride_y, const int x0_stride_y, const int Dx_stride_y, 
    const int x1_stride_z, const int x0_stride_z, const int Dx_stride_z,
    const int x0_x_offset, const int x0_y_offset, const int x0_z_offset,
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset,
    const double coef_0, const double b, const double *v0, double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + x0_x_offset;
    const int x0_y = x1_y + x0_y_offset;
    const int Dx_x = x1_x + Dx_x_offset;
    const int Dx_y = x1_y + Dx_y_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int idx_Dx_xy = Dx_y * Dx_stride_y + Dx_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    const double *Dx = _Dx + icol * DMnd_Dx;
    double *x1 = _x1 + icol * DMnd;
    
    double z_buf[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_2R];
    __shared__ double Dx_xy_plane[Y_BLK_2R][X_BLK_SIZE];
    
    bool valid_x1 = true, valid_x0 = true, valid_Dx = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx + RADIUS) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    if ((x1_x >= nx) || (x1_y >= ny + RADIUS)) valid_Dx = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + x0_z_offset;
        int Dx_z = x1_z + Dx_z_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        int idx_Dx = idx_Dx_xy + Dx_z * Dx_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 12; i++) z_buf[i] = z_buf[i + 1];
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        
        if (valid_x0) x0_xy_plane[tile_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][tile_x] = 0.0;
        
        if (valid_Dx) Dx_xy_plane[tile_y][local_x] = Dx[idx_Dx];
        else Dx_xy_plane[tile_y][local_x] = 0.0;
        
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][tile_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
            Dx_xy_plane[local_y][local_x]             = Dx[idx_Dx -     RADIUS * nx];
            Dx_xy_plane[tile_y + Y_BLK_SIZE][local_x] = Dx[idx_Dx + Y_BLK_SIZE * nx];
        }
        if (local_x < RADIUS)
        {
            x0_xy_plane[tile_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
        }
        __syncthreads(); 
        

        double res = z_buf[RADIUS] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            int r_fac = 4 * r + 1;
            res += (x0_xy_plane[tile_y][tile_x - r] + x0_xy_plane[tile_y][tile_x + r]) * cu_Lap_wt_nonorth[r_fac];
            res += (x0_xy_plane[tile_y - r][tile_x] + x0_xy_plane[tile_y + r][tile_x]) * cu_Lap_wt_nonorth[r_fac+1];
            res += (z_buf[RADIUS - r] + z_buf[RADIUS + r]) * cu_Lap_wt_nonorth[r_fac+2];
            res += (Dx_xy_plane[tile_y + r][local_x] - Dx_xy_plane[tile_y - r][local_x]) * cu_Lap_wt_nonorth[r_fac+3];
        }
            
        if (valid_x1) x1[idx_x1] = res + b * (v0[idx_x1] * x0[idx_x0]);
        __syncthreads(); 
    }
}

__global__ void Lap_nonorth_DZ_r6_kernel(
    const int nx,          const int ny,          const int nz,
    const double *_x0,     const double *_Dx,     const int stride_Dx, 
    const int DMnd,        const int DMnd_ex,     const int DMnd_Dx, 
    const int x1_stride_y, const int x0_stride_y, const int Dx_stride_y, 
    const int x1_stride_z, const int x0_stride_z, const int Dx_stride_z,
    const int x0_x_offset, const int x0_y_offset, const int x0_z_offset,
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset,
    const double coef_0, const double b, const double *v0, double *_x1
)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const int x1_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0_x = x1_x + x0_x_offset;
    const int x0_y = x1_y + x0_y_offset;
    const int Dx_x = x1_x + Dx_x_offset;
    const int Dx_y = x1_y + Dx_y_offset;
    const int idx_x0_xy = x0_y * x0_stride_y + x0_x;
    const int idx_x1_xy = x1_y * x1_stride_y + x1_x;
    const int idx_Dx_xy = Dx_y * Dx_stride_y + Dx_x;
    const int icol = blockIdx.z;
    const double *x0 = _x0 + icol * DMnd_ex;
    const double *Dx = _Dx + icol * DMnd_Dx;
    double *x1 = _x1 + icol * DMnd;
    
    double z_buf_x0[2 * RADIUS + 1];
    double z_buf_Dx[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_2R];
    
    bool valid_x1 = true, valid_x0 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx + RADIUS) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        int idx_Dx = idx_Dx_xy + RADIUS * Dx_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
        {
            z_buf_x0[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
            z_buf_Dx[RADIUS + i + 1] = Dx[idx_Dx + i * Dx_stride_z];
        }
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) 
        {
            z_buf_x0[i] = 0.0;
            z_buf_Dx[i] = 0.0;
        }
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z = x1_z + x0_z_offset;
        int Dx_z = x1_z + Dx_z_offset;
        int idx_x0 = idx_x0_xy + x0_z * x0_stride_z;
        int idx_x1 = idx_x1_xy + x1_z * x1_stride_z;
        int idx_Dx = idx_Dx_xy + Dx_z * Dx_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 12; i++)
        {
            z_buf_x0[i] = z_buf_x0[i + 1];
            z_buf_Dx[i] = z_buf_Dx[i + 1];
        }
        if (valid_x1)
        {
            z_buf_x0[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
            z_buf_Dx[2 * RADIUS] = Dx[idx_Dx + RADIUS * Dx_stride_z];
        }
        
        if (valid_x0) x0_xy_plane[tile_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][tile_x] = 0.0;
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][tile_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
        }
        if (local_x < RADIUS)
        {
            x0_xy_plane[tile_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
        }
        __syncthreads(); 

        double res = z_buf_x0[RADIUS] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            int r_fac = 4 * r + 1;
            res += (x0_xy_plane[tile_y][tile_x - r] + x0_xy_plane[tile_y][tile_x + r]) * cu_Lap_wt_nonorth[r_fac];
            res += (x0_xy_plane[tile_y - r][tile_x] + x0_xy_plane[tile_y + r][tile_x]) * cu_Lap_wt_nonorth[r_fac+1];
            res += (z_buf_x0[RADIUS - r] + z_buf_x0[RADIUS + r]) * cu_Lap_wt_nonorth[r_fac+2];
            res += (z_buf_Dx[RADIUS + r] - z_buf_Dx[RADIUS - r]) * cu_Lap_wt_nonorth[r_fac+3];
        }
        
        if (valid_x1) x1[idx_x1] = res + b * (v0[idx_x1] * x0[idx_x0]);
        __syncthreads();
    }
}

__global__ void Lap_nonorth_DX_DY_r6_kernel(
    const int nx,        const int ny,       const int nz,
    const double *_x0,   const double *_Dx1, const double *_Dx2,
    const double coef_0, const double b, const double *v0, double *_x1
)
{
    const int x0_stride_y = nx + 2 * RADIUS;
    const int x1_stride_z = nx * ny;
    const int x0_stride_z = x0_stride_y * (ny + 2 * RADIUS);
    const int Dx1_stride_z = x0_stride_y * ny;
    const int Dx2_stride_z = nx * (ny + 2 * RADIUS);
    
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int tile_x = local_x + RADIUS;
    const int tile_y = local_y + RADIUS;
    const int x1_x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int x1_y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_x1_xy  = x1_y * nx + x1_x;
    const int idx_x0_xy  = (x1_y + RADIUS) * x0_stride_y + (x1_x + RADIUS);
    const int idx_Dx1_xy = (x1_y + 0)      * x0_stride_y + (x1_x + RADIUS);
    const int idx_Dx2_xy = (x1_y + RADIUS) * nx          + (x1_x + 0);
    
    const int icol = blockIdx.z;
    int DMnd = nx * ny * nz;
    int DMnd_ex  = x0_stride_y * (ny+2*RADIUS) * (nz+2*RADIUS);
    int DMnd_Dx1 = x0_stride_y * ny * nz;
    int DMnd_Dx2 = nx * (ny+2*RADIUS) * nz;
    double *x1  = _x1  + icol * DMnd;
    const double *x0  = _x0  + icol * DMnd_ex;
    const double *Dx1 = _Dx1 + icol * DMnd_Dx1;
    const double *Dx2 = _Dx2 + icol * DMnd_Dx2;
    
    double z_buf[2 * RADIUS + 1];
    __shared__ double x0_xy_plane[Y_BLK_2R][X_BLK_2R];
    __shared__ double Dx1_xy_plane[Y_BLK_SIZE][X_BLK_2R];
    __shared__ double Dx2_xy_plane[Y_BLK_2R][X_BLK_SIZE];

    bool valid_x1 = true, valid_x0 = true, valid_Dx2 = true;
    if ((x1_x >= nx) || (x1_y >= ny)) valid_x1 = false;
    if ((x1_x >= nx + RADIUS) || (x1_y >= ny + RADIUS)) valid_x0 = false;
    if ((x1_x >= nx) || (x1_y >= ny + RADIUS)) valid_Dx2 = false;
    
    if (valid_x1)
    {
        int idx_x0 = idx_x0_xy + RADIUS * x0_stride_z;
        #pragma unroll 12
        for (int i = -RADIUS; i < RADIUS; i++)
            z_buf[RADIUS + i + 1] = x0[idx_x0 + i * x0_stride_z];
    } else {
        #pragma unroll 13
        for (int i = 0; i <= 2 * RADIUS; i++) z_buf[i] = 0.0;
    }
    
    for (int x1_z = 0; x1_z < nz; x1_z++)
    {
        int x0_z  = x1_z + RADIUS;
        int Dx1_z = x1_z + 0;
        int Dx2_z = x1_z + 0;
        int idx_x0  = idx_x0_xy  + x0_z  * x0_stride_z;
        int idx_x1  = idx_x1_xy  + x1_z  * x1_stride_z;
        int idx_Dx1 = idx_Dx1_xy + Dx1_z * Dx1_stride_z;
        int idx_Dx2 = idx_Dx2_xy + Dx2_z * Dx2_stride_z;
        
        #pragma unroll 12
        for (int i = 0; i < 2 * RADIUS; i++) z_buf[i] = z_buf[i + 1];
        
        if (valid_x0) x0_xy_plane[tile_y][tile_x] = x0[idx_x0];
        else x0_xy_plane[tile_y][tile_x] = 0.0;
        
        if (x1_x < nx + RADIUS) Dx1_xy_plane[local_y][tile_x] = Dx1[idx_Dx1];
        else Dx1_xy_plane[local_y][tile_x] = 0.0;
        
        if (valid_Dx2) Dx2_xy_plane[tile_y][local_x] = Dx2[idx_Dx2];
        else Dx2_xy_plane[tile_y][local_x] = 0.0;
        
        if (local_y < RADIUS)
        {
            x0_xy_plane[local_y][tile_x]             = x0[idx_x0 -     RADIUS * x0_stride_y];
            x0_xy_plane[tile_y + Y_BLK_SIZE][tile_x] = x0[idx_x0 + Y_BLK_SIZE * x0_stride_y];
            Dx2_xy_plane[local_y][local_x]             = Dx2[idx_Dx2 -     RADIUS * nx];
            Dx2_xy_plane[tile_y + Y_BLK_SIZE][local_x] = Dx2[idx_Dx2 + Y_BLK_SIZE * nx];
        }
        if (local_x < RADIUS)
        {
            x0_xy_plane[tile_y][local_x]             = x0[idx_x0 - RADIUS];
            x0_xy_plane[tile_y][tile_x + X_BLK_SIZE] = x0[idx_x0 + X_BLK_SIZE];
            Dx1_xy_plane[local_y][local_x]             = Dx1[idx_Dx1 - RADIUS];
            Dx1_xy_plane[local_y][tile_x + X_BLK_SIZE] = Dx1[idx_Dx1 + X_BLK_SIZE];
        }
        __syncthreads(); 
        
        if (valid_x1) z_buf[2 * RADIUS] = x0[idx_x0 + RADIUS * x0_stride_z];
        else z_buf[2 * RADIUS] = 0.0;
        
        double res = z_buf[RADIUS] * coef_0;
        #pragma unroll 6
        for (int r = 1; r <= RADIUS; r++)
        {
            int r_fac = 5 * r;
            res += (x0_xy_plane[tile_y][tile_x - r] + x0_xy_plane[tile_y][tile_x + r]) * cu_Lap_wt_nonorth[r_fac];
            res += (x0_xy_plane[tile_y + r][tile_x] + x0_xy_plane[tile_y - r][tile_x]) * cu_Lap_wt_nonorth[r_fac+1];
            res += (z_buf[RADIUS - r] + z_buf[RADIUS + r]) * cu_Lap_wt_nonorth[r_fac+2];
            res += (Dx1_xy_plane[local_y][tile_x + r] - Dx1_xy_plane[local_y][tile_x - r]) * cu_Lap_wt_nonorth[r_fac+3];
            res += (Dx2_xy_plane[tile_y + r][local_x] - Dx2_xy_plane[tile_y - r][local_x]) * cu_Lap_wt_nonorth[r_fac+4];
        }
        
        if (valid_x1) x1[idx_x1] = res + b * (v0[idx_x1] * x0[idx_x0]);
        __syncthreads(); 
    }
}


// ==================== Other kernels used in CheFSI ====================

__global__ void Vnl_SpMV_scale_alpha_kernel(int alen, double *alpha_scale, double *alpha)
{
    int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx1 = blockIdx.y * alen + idx0;
    if (idx0 < alen) alpha[idx1] *= alpha_scale[idx0];
}


__global__ void update_Ynew_by_X_kernel(int XY_size, double vscal1, double vscal2, double *X, double *Ynew)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < XY_size) 
    {
        Ynew[tid] *= vscal1;
        Ynew[tid] -= vscal2 * X[tid];
    }
}
