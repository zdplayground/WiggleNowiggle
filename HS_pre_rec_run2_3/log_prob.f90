! made on 08/24/2016, get subroutine of lnlike function in mcmc code
! Use f2py -m lnprob_module -h lnprob_module.pyf log_prob.f90 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 lnprob_module.pyf log_prob.f90 to generate the module
!
subroutine match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    implicit none
    integer:: dim_theta, dim_params, count, i
    double precision:: theta(dim_theta), params_indices(dim_params), fix_params(dim_params)
!f2py intent(in):: theta, params_indices, fix_params
    double precision:: params_array(dim_params)
!f2py intent(out):: params_array

    count = 1  ! be very careful that the starting value is different from Python's. It's 1 in fortran!
    do i=1, dim_params
        if (params_indices(i) == 1) then
            params_array(i) = theta(count)
            count = count + 1
        else
            params_array(i) = fix_params(i)
        endif
    end do
    return
   
end subroutine 

subroutine cal_Pk_model(Pk_linw, Pk_sm, k_t, mu_t, sigma_xy, sigma_z, b_0, b_scale, norm_gf, Pk_model, dim_kt)
    implicit none
    integer:: dim_kt, i
    double precision, dimension(dim_kt):: Pk_linw, Pk_sm, k_t, mu_t
    double precision:: sigma_xy, sigma_z, b_0, b_scale, norm_gf
    double precision:: Pk_model(dim_kt), exp_term
!f2py intent(in):: Pk_linw, Pk_sm, k_t, mu_t, sigma_xy, sigma_z, b_0, b_scale, norm_gf
!f2py intent(out):: Pk_model
    do i=1, dim_kt
        exp_term = exp(k_t(i)*k_t(i) *(mu_t(i)*mu_t(i)*(sigma_xy+sigma_z)*(sigma_xy-sigma_z)-sigma_xy**2.0)/2.0)
        Pk_model(i) = (Pk_linw(i)-Pk_sm(i))* exp_term * (1.0+b_scale * k_t(i)**2.0) * (b_0 * norm_gf)**2.0
    enddo
    return 
end subroutine 

subroutine lnprior(theta, params_indices, fix_params, lp, dim_theta, dim_params)
    implicit none
    integer:: dim_theta, dim_params
    double precision:: theta(dim_theta), params_indices(dim_params), fix_params(dim_params)
    double precision:: lp
    double precision:: params_array(dim_params), alpha_1, alpha_2, sigma_xy, sigma_z, b_0, b_scale
    
!f2py intent(in):: theta, params_indices, fix_params
!f2py intent(out):: lp
    call match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    alpha_1 = params_array(1)
    alpha_2 = params_array(2)
    sigma_xy = params_array(3)
    sigma_z = params_array(4)
    b_0 = params_array(5)
    b_scale = params_array(6)

    if (alpha_1>-1.d-7 .and. alpha_1<1.1 .and. alpha_2>-1.d-7 .and. alpha_2<1.1 .and. sigma_xy>-1.d-7 .and. sigma_xy<15.0 &
        .and. sigma_z>-1.d-7 .and. sigma_z<15.0 .and. b_0>0. .and. b_0<6.0 .and. b_scale>-100.0 .and. b_scale<100.0) then
        lp = 0.d0
    else
        lp = -1.d30  ! return a negative infinitely large number
    endif
    !print*, lp
    return
end subroutine 
    

