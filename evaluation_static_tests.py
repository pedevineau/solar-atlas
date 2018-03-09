'''
author: Pierre-Etienne Devineau
SOLARGIS S.R.O.

To measure the relative efficiency of each cloud test
'''


def corr2test(test_1, test_2):
    # expect list
    # test_1 = test_1.flatten()
    # test_2 = test_2.flatten()
    assert len(test_1) == len(test_2), 'the two lists have different sizes'
    score = (1.*sum(test_1 == test_2)) / len(test_1)
    return score


def compute_common_scores(*args):
    # expect list
    n = len(args)
    common = args[0]
    for k in range(1, n):
        common = (args[k] | common)
    score = []
    for l in range(n):
        score.append(corr2test(common, args[l]))
    return score


def compute_unique_scores(*args):
    # expect list
    n = len(args)
    from numpy import zeros
    common = args[0]
    for k in range(1, n):
        common = (args[k] | common)
    scores = []
    for k in range(n):
        for l in range(n):
            diff = zeros(len(args[0]), dtype=bool)
            if k != l:
                diff = diff | (args[k] ^ args[l])
        scores.append(corr2test(diff, common))
    return scores


def visualize_regular_cloud_tests():
    '''
    Show the clouds detected by regular test above lands (does not include the visible cloud test
     and the unstable snowy test), and print the "individual score" of each test (=the ratio of clouds
     which have been detected only by this test)
    :return:
    '''
    from static_tests import dawn_day_test, cli_water_cloud_test, cli_stability, gross_cloud_test, thin_cirrus_test, \
        typical_static_classifier, epsilon_transparent_cloud_test
    from numpy import zeros_like
    from utils import visualize_map_time, typical_bbox
    zen, lands, cli_mu, cli_var, cli_epsilon, vis, lir, fir, lir_forecast, fir_forecast = typical_static_classifier(
        bypass=True)
    common_condition = lands & dawn_day_test(zen)
    cli_condition = (cli_water_cloud_test(cli_mu) & cli_stability(cli_var))
    epsilon_condition = epsilon_transparent_cloud_test(cli_epsilon)
    gross_condition = gross_cloud_test(lir, lir_forecast)
    thin_condition = thin_cirrus_test(lands, lir, fir, lir_forecast, fir_forecast)

    print 'CONDITIONS: GROSS, CLI, EPSILON, THIN'
    print 'COMMON SCORES FOR LAND DAWN-DAY PRIMARY CLOUD CLASSIFICATION'
    print compute_common_scores(
        gross_condition[common_condition],
        cli_condition[common_condition],
        epsilon_condition[common_condition],
        thin_condition[common_condition],
    )
    print 'INDIVIDUAL SCORES FOR LAND DAWN-DAY PRIMARY CLOUD CLASSIFICATION'
    print compute_unique_scores(
        gross_condition[common_condition],
        cli_condition[common_condition],
        epsilon_condition[common_condition],
        thin_condition[common_condition],
    )
    to_return = zeros_like(cli_condition)
    to_return[gross_condition] = 1
    to_return[cli_condition] = 2
    to_return[epsilon_condition] = 3
    to_return[thin_condition] = 4
    visualize_map_time(to_return, typical_bbox(), vmin=0, vmax=4)
    visualize_map_time(gross_condition, typical_bbox(), vmin=0, vmax=1)
    visualize_map_time(cli_condition, typical_bbox(), vmin=0, vmax=1)
    visualize_map_time(epsilon_condition, typical_bbox(), vmin=0, vmax=1)
    visualize_map_time(thin_condition, typical_bbox(), vmin=0, vmax=1)


if __name__ == '__main__':
    visualize_regular_cloud_tests()
