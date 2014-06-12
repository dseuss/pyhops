module hint
use system
use todeint, only: zintegrate_rk4
implicit none
include 'mkl_spblas.fi'

contains

subroutine calc_trajectory_lin(t_length, t_steps, dim_hs, nr_eq, psi0, psi, &
         lin_nnz, lin_i, lin_j, lin_a)
   implicit none
   real(dp), intent(in)     :: t_length
   integer(ki), intent(in)  :: t_steps
   integer(ki), intent(in)  :: dim_hs
   integer(ki), intent(in)  :: nr_eq
   complex(dp), intent(in)  :: psi0(nr_eq)
   complex(dp), intent(out) :: psi(dim_hs, t_steps)

   ! Linear propagator as sparse matrix
   integer(ki), intent(in) :: lin_nnz
   integer(ki), intent(in) :: lin_i(nr_eq + 1)
   integer(ki), intent(in) :: lin_j(lin_nnz)
   complex(dp), intent(in) :: lin_a(lin_nnz)

   complex(dp), allocatable :: dummy(:, :)

   allocate(dummy(0, 2*t_steps))
   dummy = 0._dp

   psi = zintegrate_rk4(rhs, t_length, t_steps, dummy, psi0, dim_hs)
contains

   subroutine rhs(y, f, ydot)
      implicit none
      complex(dp), intent(in) :: y(:)
      complex(dp), intent(in) :: f(:)
      complex(dp), intent(out) :: ydot(:)

      character, parameter :: trans = 'n'

      call mkl_cspblas_zcsrgemv(trans, nr_eq, lin_a, lin_i, lin_j, y, ydot)
      ! TODO Add noise terms
   end subroutine rhs

end subroutine calc_trajectory_lin

end module hint
