def corr2test(test_1, test_2):
    test_1 = test_1.flatten()
    test_2 = test_2.flatten()
    assert len(test_1) == len(test_2), 'the two arrays have different sizes'
    score = (1.*sum(test_1 == test_2)) / len(test_1)
    print score
    return score


def unique_score_one_array(*args):
    from numpy import asarray, zeros_like
    n = len(args)
    common = zeros_like(args[0], dtype=bool)
    for k in range(n):
        common = (common | asarray(args[k]).flatten())
    score = []
    for l in range(n):
        score.append(corr2test(common, asarray(args[l]).flatten()))
    return score


if __name__ == '__main__':
    from static_tests import *
    from numpy import zeros_like
    from utils import visualize_map_time, typical_bbox
    zen, lands, cli_mu, cli_var, cli_epsilon, vis, lir, fir, lir_forecast, fir_forecast = typical_static_classifier(bypass=False)
    common_condition = lands & dawn_day_test(zen)
    cli_condition = (cli_water_cloud_test(cli_mu) & cli_stability(cli_var))
    epsilon_condition = (cli_water_cloud_test(cli_mu) & cli_stability(cli_var))
    gross_condition = gross_cloud_test(lir, lir_forecast)
    thin_condition = thin_cirrus_test(lands, lir, fir, lir_forecast, fir_forecast)
    scores = unique_score_one_array(
        common_condition & cli_condition,
        common_condition & epsilon_condition,
        common_condition & gross_condition,
        common_condition & thin_condition
    )

    print 'SCORES FOR LAND DAWN-DAY PRIMARY CLOUD CLASSIFICATION'
    print 'CONDITIONS: CLI, EPSILON, GROSS, THIN'

    to_return = zeros_like(cli_condition)
    to_return[gross_condition] = 1
    to_return[cli_condition] = 2
    to_return[epsilon_condition] = 3
    to_return[thin_condition] = 4
    visualize_map_time(to_return, typical_bbox())
