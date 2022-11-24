# noqa
# type:ignore


def dict_merge(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                dict_merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def pytest_generate_tests(metafunc):
    if metafunc.cls is None:
        return
    if not hasattr(metafunc.cls, 'test_scenarios'):
        return
    test_scenarios = metafunc.cls.test_scenarios
    idlist = []
    names = []
    values = []
    defaults = test_scenarios.get('DEFAULTS', {})
    for scenario in test_scenarios:
        if scenario == 'DEFAULTS':
            continue
        test_scenario = dict_merge(dict(defaults), test_scenarios[scenario])

        fixed_fixtures = []
        param_fixtures = []
        length = None
        for fixture_name in test_scenario:
            if fixture_name == 'IDS':
                continue
            fixture_value = test_scenario[fixture_name]
            if isinstance(fixture_value, list):
                if len(fixture_value) == 1:
                    # Treat single-item list as fixed fixture
                    fixed_fixtures.append(fixture_name)
                else:
                    # We will parametrize this fixture
                    param_fixtures.append(fixture_name)
                    if length is not None and len(fixture_value) != length:
                        raise RuntimeError(
                            'All parameterized fixtures must be lists of equal length'
                        )
                    length = len(fixture_value)
            else:
                # This one will be kept fixed
                fixed_fixtures.append(fixture_name)

        if len(param_fixtures) == 0:
            # We generate a single test per scenario
            # Test id is just scenario name
            idlist.append(scenario)
            names = fixed_fixtures
            values.append(
                [
                    test_scenario[x][0]
                    if isinstance(test_scenario[x], list) else test_scenario[x]
                    for x in fixed_fixtures
                ]
            )

        else:
            # We generate multiple tests per scenario

            # Append id or a number to scenario name for each param
            if 'IDS' in test_scenario:
                ids = [f'{scenario}-{x}' for x in test_scenario['IDS']]
            else:
                ids = [f'{scenario}-{n}' for n in range(length)]
            idlist.extend(ids)
            for i in range(length):
                names = fixed_fixtures + param_fixtures
                vals = []
                for name in names:
                    if name in fixed_fixtures:
                        if isinstance(test_scenario[name], list):
                            assert len(test_scenario[name]) == 1
                            # Pull out the single item
                            vals.append(test_scenario[name][0])
                        else:
                            # print('u', test_scenario[name])
                            vals.append(test_scenario[name])
                    else:
                        # Parametrized fixture
                        vals.append(test_scenario[name][i])
                values.append(vals)

    metafunc.parametrize(names, values, ids=idlist, scope='class')
