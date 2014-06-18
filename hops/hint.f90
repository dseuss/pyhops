module hint
use system
use todeint, only: zintegrate_rk4
implicit none
include 'mkl_spblas.fi'

character, parameter :: TRANS = 'n'


contains

subroutine calc_trajectory_lin(&
         t_length, t_steps, dim_hs, nr_aux_states, nr_noise, psi0, psi, &
         lin_nnz, lin_i, lin_j, lin_a, &
         noise_nnz, noise_i, noise_j, noise_a, noise_c, noise)
   implicit none
   real(dp), intent(in)       :: t_length
   integer(ki), intent(in)    :: t_steps
   integer(ki), intent(in)    :: dim_hs
   integer(ki), intent(in)    :: nr_aux_states
   integer(ki), intent(in)    :: nr_noise
   complex(dp), intent(inout) :: psi0(nr_aux_states)
   complex(dp), intent(out)   :: psi(dim_hs, t_steps)

   ! Linear propagator as sparse matrix
   integer(ki), intent(in) :: lin_nnz
   integer(ki), intent(in) :: lin_i(nr_aux_states + 1)
   integer(ki), intent(in) :: lin_j(lin_nnz)
   complex(dp), intent(in) :: lin_a(lin_nnz)

   ! Propagator proportional to ~ Z_t and Z_t
   integer(ki), intent(in) :: noise_nnz
   integer(ki), intent(in) :: noise_i(nr_aux_states + 1)
   integer(ki), intent(in) :: noise_j(noise_nnz)
   integer(ki), intent(in) :: noise_a(noise_nnz)
   complex(dp), intent(in) :: noise_c(noise_nnz)

   complex(dp), intent(in) :: noise(nr_noise, 2 * t_steps)

   psi = zintegrate_rk4(rhs, t_length, t_steps, noise, psi0, dim_hs)
contains

   subroutine rhs(y, f, ydot)
      implicit none
      complex(dp), intent(in) :: y(:)
      complex(dp), intent(in) :: f(:)
      complex(dp), intent(out) :: ydot(:)

      call mkl_cspblas_zcsrgemv(TRANS, nr_aux_states, lin_a, lin_i, lin_j, y, ydot)
      if (nr_noise /= 0) then
         call timedep_csr_matvec(noise_i, noise_j, noise_a, f, y, ydot, &
               noise_c)
      endif
   end subroutine rhs

end subroutine calc_trajectory_lin


subroutine timedep_csr_matvec(i_csr, j_csr, a_csr, f, x, y, c_csr)
   ! Compute y += A(f).x for CSR matrix A and dense vectors x, y
   implicit none
   integer(ki), intent(in)           :: i_csr(0:)
   integer(ki), intent(in)           :: j_csr(0:)
   integer(ki), intent(in)           :: a_csr(0:)
   complex(dp), intent(in)           :: f(0:)
   complex(dp), intent(in)           :: x(0:)
   complex(dp), intent(inout)        :: y(0:)
   complex(dp), intent(in), optional :: c_csr(0:)

   integer :: i, j

   if (present(c_csr)) then
      do i = 0, size(i_csr) - 2
         do j = i_csr(i), i_csr(i+1) - 1
            y(i) = y(i) + c_csr(j) * f(a_csr(j)) * x(j_csr(j))
         end do
      end do

   else
      do i = 0, size(i_csr) - 2
         do j = i_csr(i), i_csr(i+1) - 1
            y(i) = y(i) + f(a_csr(j)) * x(j_csr(j))
         end do
      end do
   end if
end subroutine timedep_csr_matvec

end module hint
