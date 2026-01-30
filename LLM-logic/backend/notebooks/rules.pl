% Facts
ucsc_undergrad(ucsc_undergrad).
files_plan_within_max_enrollment_quarters(files_plan_within_max_enrollment_quarters).
declared(declared_s, declared_m).
completes_requirements(completes_requirements_s, completes_requirements_m).
at_least_40_unique_upper_div_credits(at_least_40_unique_upper_div_credits_s, at_least_40_unique_upper_div_credits_m).

% Rule
may_double_major(S) :-
    ucsc_undergrad(S),
    files_plan_within_max_enrollment_quarters(S),
    forall(declared(S,M),
           (completes_requirements(S,M),
            at_least_40_unique_upper_div_credits(S,M))).