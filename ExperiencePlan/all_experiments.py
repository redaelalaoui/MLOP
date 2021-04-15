from ExperiencePlan.experience_plan_function import experience_plan


def all_experiments(data_size):
    experience_all_results, _, _ = experience_plan([data_size], naive_vs_reduced=(True, True), price_vs_pp=(True, True),
                                   stand_vs_nostand=(True, True), iso_vs_aniso=(True, True),
                                   kernels=(True, True, True))

    treated_results = experience_all_results
    treated_results = treated_results.sort_values('MSE')

    treated_results.to_excel(f'Results/all_possibilities_{data_size}.xlsx')


if __name__ == "__main__":
    all_experiments(1000)
