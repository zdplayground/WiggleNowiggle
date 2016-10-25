! Copy the code from pre_log_prob_rsd.f90, modify it to get subroutine of lnlike function in redshift space for post-reconstruction.
! The likelihood function is referenced from Zvonimir Vlah. -- 10/14/2016.
! Use f2py -m post_lnprob_module_rsd_ZV -h post_lnprob_module_rsd_ZV.pyf post_log_prob_rsd_ZV.f90 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 post_lnprob_module_rsd_ZV.pyf post_log_prob_rsd_ZV.f90 to generate the module
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

subroutine cal_Pk_model(Pk_linw, Pk_sm, k_t, mu_t, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4,&
                        norm_gf, Pk_model, dim_kt)
    implicit none
    integer:: dim_kt, i
    double precision, dimension(dim_kt):: Pk_linw, Pk_sm, k_t, mu_t
    double precision:: sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4, norm_gf
    double precision:: sm_term, f_sm, t1, t2, t3, Pk_model(dim_kt)
!f2py intent(in):: Pk_linw, Pk_sm, k_t, mu_t, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0, b_scale0, b_scale2, b_scale4, norm_gf
!f2py intent(out):: Pk_model
    do i=1, dim_kt
        sm_term = 1.d0-exp(-0.25*(k_t(i)*sigma_sm)**2.d0)
        f_sm = f*sm_term
        t1 = b_0**2.d0*(1.d0+b_scale0*k_t(i)**2.d0)*exp(-(k_t(i)*sigma_0)**2.d0/2.d0)
        t2 = 2.d0*f_sm*b_0*mu_t(i)**2.d0 * (1.d0+b_scale2 *k_t(i)**2.d0)*exp(-(k_t(i)*sigma_2)**2.d0/2.d0)
        t3 = f_sm*f_sm*mu_t(i)**4.d0 *(1.d0+b_scale4*k_t(i)**2.d0)*exp(-(k_t(i)*sigma_4)**2.d0/2.d0)
        Pk_model(i) = (Pk_linw(i)-Pk_sm(i))*(t1+t2+t3)* norm_gf**2.d0
    enddo
    return 
end subroutine 

subroutine lnprior(theta, params_indices, fix_params, lp, dim_theta, dim_params)
    implicit none
    integer:: dim_theta, dim_params
    double precision:: theta(dim_theta), params_indices(dim_params), fix_params(dim_params)
    double precision:: lp
    double precision:: params_array(dim_params), alpha_1, alpha_2, sigma_0, sigma_2, sigma_4, sigma_sm, f, b_0
    double precision:: b_scale0, b_scale2, b_scale4
    
!f2py intent(in):: theta, params_indices, fix_params
!f2py intent(out):: lp
    call match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    alpha_1 = params_array(1)
    alpha_2 = params_array(2)
    sigma_0 = params_array(3)
    sigma_2 = params_array(4)  ! remove abs() and set them positive in params_array before passed by.--10/06/2016
    sigma_4 = params_array(5)
    sigma_sm = params_array(6)  ! sigma_sm is not a fitting parameter.
    f = params_array(7)
    b_0 = params_array(8)
    b_scale0 = params_array(9)
    b_scale2 = params_array(10)        
    b_scale4 = params_array(11) 

    if (alpha_1>-1.d-7 .and. alpha_1<1.1 .and. alpha_2>-1.d-7 .and. alpha_2<1.1 .and. sigma_0>-1.d-7 .and. sigma_0<100.0 &
        .and. sigma_2>-1.d-7 .and. sigma_2<1.d4 .and. sigma_4>-1.d-7 .and. sigma_4<1.d4 .and. b_0>0. .and. b_0<6.0 &
        .and. b_scale0>-1.d4 .and. b_scale0<1.d4 .and. b_scale2>-1.d4 .and. b_scale2<1.d4 .and. b_scale4>-1.d4 &
        .and. b_scale4<1.d4 .and. f > 0.0 .and. f < 4.0) then
        lp = 0.d0
    else
        lp = -1.d30  ! return a negative infinitely large number
    endif
    !print*, lp
    return
end subroutine 
    

