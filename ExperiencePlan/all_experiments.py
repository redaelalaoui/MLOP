from typing import NoReturn

from ExperiencePlan.experience_plan_function import experience_plan


def all_experiments(data_size: int) -> NoReturn:
    experience_all_results, _, _ = experience_plan([data_size], naive_vs_reduced=(False, True), price_vs_pp=(False, True),
                                   stand_vs_nostand=(True, True), iso_vs_aniso=(False, True),
                                   kernels=(True, False, False))

    treated_results = experience_all_results
    treated_results = treated_results.sort_values('MSE')

    treated_results.to_excel(f'Results/all_possibilities_{data_size}.xlsx')


if __name__ == "__main__":
    all_experiments(1000)
