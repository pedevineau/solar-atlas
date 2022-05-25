def p_ground(parameter):
    assert parameter in [
        "cli",
        "var-cli",
        "pos-var-ndsi",
        "neg-var-ndsi",
        "cold-anomaly",
        "ndsi",
    ], "unknown parameter"
    switch(parameter)


def bayesian_inference():
    global cloud_stats, snow_stats, ground_stats
    proba_ground_given_inputs = (
        p_ground("cli")
        * p_ground("neg-var-ndsi")
        * p_ground("pos-var-ndsi")
        * p_ground("cold-anomaly")
        * p_ground("var-cli")
        * p_ground("ndsi")
    )

    proba_snow_given_inputs = (
        p_snow("cli")
        * p_snow("neg-var-ndsi")
        * p_snow("pos-var-ndsi")
        * p_snow("cold-anomaly")
        * p_snow("var-cli")
        * p_snow("ndsi")
    )

    proba_cloud_given_inputs = (
        p_cloud("cli")
        * p_cloud("neg-var-ndsi")
        * p_cloud("pos-var-ndsi")
        * p_cloud("cold-anomaly")
        * p_cloud("var-cli")
        * p_cloud("ndsi")
    )

    cloud_per_ground = (
        cloud_stats
        / ground_stats
        * proba_cloud_given_inputs
        / proba_ground_given_inputs
    )
    ground_per_snow = (
        ground_stats / snow_stats * proba_ground_given_inputs / proba_snow_given_inputs
    )

    snow = 1.0 / (1 + cloud_per_ground * ground_per_snow + ground_per_snow)
    ground = snow * ground_per_snow
    cloud = ground * cloud_per_ground
    return ground, snow, cloud
