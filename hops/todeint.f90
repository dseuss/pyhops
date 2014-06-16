! Module for integration of time-dependent ODEs, where the time dependent
! function is only known on a finite time-grid (i.e. a realization of a
! stochastic process calculated using Fourier-filter).

module todeint
use system
implicit none

! RUNGE-KUTTA PARAMETERS
integer, parameter :: &
      RK_COPIES = 4, &     ! Number of copies needed
      RK_1 = 1, &
      RK_2 = 2, &
      RK_3 = 3, &
      RK_4 = 4

contains

! Integrates the ODE dy/dt = rhs(y) using the fixed time step Runge-Kutta method
! of 4th order. Here, the rhs may be a time-dependent function through the time-
! dependent parameter f, which is passed to the rhs-method and processed by the
! latter.
!
! It calculates on the y0-memory passed in.
!
! TODO output_crop can be array of ints
function zintegrate_rk4(rhs, t_length, t_steps, f, y, output_crop) result(res)
   implicit none
   interface
      subroutine rhs(y, f, ydot)
         use system, only: dp
         complex(dp), intent(in)  :: y(:)        ! y(:, t)
         complex(dp), intent(in)  :: f(:)        ! f(:, t)
         complex(dp), intent(out) :: ydot(:)     ! ydot(:, t)
      end subroutine rhs
   end interface

   real(dp), intent(in)       :: t_length
   integer(ki), intent(in)    :: t_steps
   complex(dp), intent(in)    :: f(:, :)
   complex(dp), intent(inout) :: y(:)
   integer(ki), intent(in)    :: output_crop
   complex(dp)                :: res(output_crop, t_steps)
   !----------------------------------------------------------------------------
   real(dp) :: dt
   complex(dp) :: yrk(size(y), RK_COPIES)
   integer(ki) :: t

   ! TODO ERROR CHECK shape(f) = [nr_procs, 2*tSteps]

   dt = t_length / (t_steps - 1)
   res(1:output_crop, 1) = y(1:output_crop)

   do t = 2, t_steps
      call rhs(y(:), f(:, 2*t - 2), yrk(:, RK_1))
      yrk(:, RK_1) = dt * yrk(:, RK_1)

      call rhs(y(:) + .5 * yrk(:, RK_1), f(:, 2*t - 1), yrk(:, RK_2))
      yrk(:, RK_2) = dt * yrk(:, RK_2)

      call rhs(y(:) + .5 * yrk(:, RK_2), f(:, 2*t - 1), yrk(:, RK_3))
      yrk(:, RK_3) = dt * yrk(:, RK_3)

      call rhs(y(:) + yrk(:, RK_3), f(:, 2*t), yrk(:, RK_4))
      yrk(:, RK_4) = dt * yrk(:, RK_4)

      y(:) = y(:) &
            + 1./6. * yrk(:, RK_1) &
            + 2./6. * yrk(:, RK_2) &
            + 2./6. * yrk(:, RK_3) &
            + 1./6. * yrk(:, RK_4)
      res(:, t) = y(1:output_crop)
   end do
end function zintegrate_rk4

end module todeint
