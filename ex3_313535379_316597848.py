import pandas as pd
import math
from itertools import permutations
from math import comb
import decimal

decimal.getcontext().prec = 100


########## Part A ###############


def opt_bnd(data, k: int, years: list) -> dict:
    # returns the optimal bundle of cars for that k and list of years and their total cost.
    cost = 0
    bundle = []
    data['picked'] = [0] * len(data)
    brands = ['ford', 'bmw', 'kia', 'vw', 'ferrari']
    omega = [list(zip(x, years)) for x in permutations(brands, len(years))]

    for i in range(k):
        global_best_bundle = None
        global_v = float('inf')

        for sigma in omega:
            v_sigma = 0
            best_bundle_sigma = []
            for car_year in sigma:
                valid_cars = data[
                    (data['brand'] == car_year[0]) & (data['year'] == car_year[1]) & (data['picked'] == 0)]
                car = valid_cars[valid_cars['value'] == valid_cars.value.min()]
                v_sigma += car['value'].tolist()[0]
                best_bundle_sigma.append(car['id'].tolist()[0])
            if v_sigma < global_v:
                global_best_bundle = best_bundle_sigma
                global_v = v_sigma

        # update picked column & add to bundle
        for picked_car in global_best_bundle:
            data.loc[data['id'] == picked_car, 'picked'] = 1
            bundle.append(picked_car)

        cost += global_v

    return {"cost": cost, "bundle": bundle}


def proc_vcg(data, k, years):
    # runs the VCG procurement auction
    cars_p = {}
    bundle = opt_bnd(data, k, years)
    cost = bundle['cost']
    cars = bundle['bundle']
    for car in cars:
        cars_p[car] = abs(
            (cost - int(data.loc[data['id'] == car, 'value'])) -
            opt_bnd(data.drop(data[data.id == car].index), k, years)[
                'cost'])
    return cars_p


########## Part B ###############
def extract_data(brand, year, size, data):
    # extract the specific data for that type
    return data[(data.brand == brand) & (data.year == year) & (data.engine_size == size)]['value'].tolist()


def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n // 2 - 1:n // 2 + 1]) / 2.0, s[n // 2])[n % 2] if n else None


class Type:
    cars_num = 0
    buyers_num = 0

    def __init__(self, brand, year, size, data):
        self.data = extract_data(brand, year, size, data)

    def avg_buy(self):
        # runs a procurement vcg auction for buying cars_num cars on the given self.data.
        # returns the average price paid for a winning car.
        return sorted(self.data)[self.cars_num]

    def cdf(self, x):
        # return F(x) for the histogram self.data
        data = sorted(self.data)
        unique_data = sorted(list(set(data)))
        if x >= data[-1]:
            return 1
        if x < data[0]:
            return 0

        normalizer = len(data)
        count = 0
        data_hist = {}
        for p in data:
            count += 1 / normalizer
            data_hist[p] = count

        for i in range(len(unique_data)):
            if unique_data[i] == x:
                return data_hist[x]
            if unique_data[i] > x:
                return data_hist[unique_data[i]] * (x - unique_data[i - 1]) / (unique_data[i] - unique_data[i - 1]) \
                       + data_hist[unique_data[i - 1]] * (unique_data[i] - x) / (unique_data[i] - unique_data[i - 1])

    def os_cdf(self, r, n, x):
        # The r out of n order statistic CDF
        return sum([comb(n, j) * (self.cdf(x) ** j) * ((1 - self.cdf(x)) ** (n - j)) for j in range(r, n + 1)])

    def exp_rev(self):
        # returns the expected revenue in future auction for cars_num items and buyers_num buyers
        i = 0
        s_i = 1 - self.os_cdf(self.buyers_num - self.cars_num, self.buyers_num, i)
        s = s_i
        while s_i > 0:
            i += 1
            s_i = 1 - self.os_cdf(self.buyers_num - self.cars_num, self.buyers_num, i)
            s += s_i
        return s * self.cars_num

    def exp_rev_median(self, n):
        reserved_price = int(median(self.data))
        p = self.cdf(reserved_price)
        return n * (p ** (n - 1)) * (1 - p) * reserved_price + sum(
            [1 - self.os_cdf(n - 1, n, reserved_price) for r in range(reserved_price)]) + sum(
            [1 - self.os_cdf(n - 1, n, r) for r in range(reserved_price, max(self.data) + 1)])

    ########## Part C ###############

    def reserve_price(self):
        return sum([1 - self.os_cdf(self.buyers_num - self.cars_num + 1, self.buyers_num, x) for x in
                    range(0, max(self.data) + 1)])
