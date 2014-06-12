module system
implicit none

integer, parameter, public :: &
      sp = kind(1.0), &
      dp = kind(1.d0), &
      ki = kind(1)
complex(dp), parameter, public :: &
      ii = (0._dp, 1._dp)

end module system
