import py.test
import numpy as np
import gf
from numpy.testing import assert_equal

def test_primpoly():
    primpoly = 11
    right_primpoly = np.array([[7, 2],
                               [1, 4],
                               [3, 3],
                               [2, 6],
                               [6, 7],
                               [4, 5],
                               [5, 1]])
    assert_equal(right_primpoly, gf.gen_pow_matrix(primpoly))
    

    
def test_add():
    X = np.array([[1, 5],
                  [17, 9],
                  [23, 5]])

    Y = np.array([[2, 3],
                  [5, 4],
                  [3, 21]])
    
    right_sum = np.array([[ 3,  6],
                          [20, 13],
                          [20, 16]])
    
    assert_equal(right_sum, gf.add(X, Y))
    
    
def test_sum():
    X = np.array([[1, 5],
              [17, 9],
              [23, 5]])
    right_sum_1 = np.array([[7, 9]])
    right_sum_2 = np.array([[ 4],
                            [24],
                            [18]])
    assert_equal(right_sum_1, gf.sum(X, axis=0))
    assert_equal(right_sum_2, gf.sum(X, axis=1))
    
    
def test_prod():
    X = np.array([[1, 5],
                  [14, 9],
                  [13, 5]])

    Y = np.array([[2, 3],
                  [5, 4],
                  [3, 11]])
    
    pm = np.array([[12,  2],
                   [13,  4],
                   [ 0,  8],
                   [14,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8],
                   [15,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8],
                   [ 0,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8]])
    
    right_prod = np.array([[4, 8],
                           [8, 4],
                           [8, 8]])
    
    assert_equal(right_prod, gf.prod(X, Y, pm))
    
    
def test_divide():
    X = np.array([[1, 5],
                  [14, 2],
                  [13, 5]])

    Y = np.array([[2, 3],
                  [2, 4],
                  [2, 11]])
    
    pm = np.array([[12,  2],
                   [13,  4],
                   [ 0,  8],
                   [14,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8],
                   [15,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8],
                   [ 0,  1],
                   [ 0,  2],
                   [ 0,  4],
                   [ 0,  8]])
    
    right_div = np.array([[4, 8],
                           [4, 4],
                           [4, 8]])
    
    assert_equal(right_div, gf.divide(X, Y, pm))    
    
    
def test_linsolve():
    pm = gf.gen_pow_matrix(37)
    A = np.array([[30, 15, 13,  2, 17, 10, 27, 16,  7, 12],
               [ 1, 15, 30,  2, 17,  4, 19,  9,  1, 11],
               [29, 26,  7,  1, 27,  2,  2, 15, 15, 18],
               [29, 12,  5,  6, 26, 18, 23, 15, 24,  4],
               [10, 24, 22, 19,  3, 31, 18, 29, 24, 30],
               [15,  8,  7,  3,  8, 22, 13,  1, 16,  4],
               [19, 25, 31,  3, 14, 29, 27,  1, 29, 12],
               [ 8, 25, 20, 17,  5, 13, 31,  7, 23,  1],
               [14, 15,  7, 26, 14,  3, 31, 16,  5,  7],
               [19, 28,  9, 11, 30,  7, 25,  2,  3, 26]])
    b = np.array([19, 2, 3, 11, 8, 27, 9, 4, 21, 5])
    right_answer = np.array([ 6, 22, 14,  8, 20, 21, 20, 23, 30, 27])
    
    assert_equal(right_answer, gf.linsolve(A, b, pm))
    
    
def test_minpoly():
    pm = gf.gen_pow_matrix(37)
    x = np.array([19, 25, 31, 3, 14, 29])
    right_answer = (np.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
                    np.array([ 3,  5,  6,  8,  9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25, 26,
                               29, 30, 31]))
    result = gf.minpoly(x, pm)
    assert_equal(right_answer[0], result[0])
    assert_equal(right_answer[1], result[1])
    
    
def test_polyval():
    pm = gf.gen_pow_matrix(37)
    p = np.array([11, 29, 26, 31, 3])
    x = np.array([19, 25, 31, 3, 14, 29])
    right_answer = np.array([ 3, 12, 19, 26, 22,  1])
    
    assert_equal(right_answer, gf.polyval(p, x, pm))
    
    
def test_polyprod():
    pm = gf.gen_pow_matrix(37)
    p1 = np.array([11, 29, 26, 31, 3])
    p2 = np.array([19, 25, 31, 3, 14, 29])
    right_answer = np.array([28, 25, 29, 30, 27, 17, 14,  1, 27,  2])
    
    assert_equal(right_answer, gf.polyprod(p1, p2, pm))
    
    
def test_polydivmod():
    pm = gf.gen_pow_matrix(37)
    p1 = np.array([13, 11, 29, 26, 31, 3])
    p2 = np.array([19, 25, 31, 3, 14, 29])
    right_answer = (np.array([21]), np.array([16, 23,  0, 23,  6]))
    result = gf.polydivmod(p1, p2, pm)
    assert_equal(right_answer[0], result[0])
    assert_equal(right_answer[1], result[1])


def test_euclid():
    pm = gf.gen_pow_matrix(37)
    p1 = np.array([ 2, 14, 22, 23,  8, 17, 31, 11, 26,  3])
    p2 = np.array([31, 23, 30, 31, 11,  9])
    right_answer = (np.array([24,  8, 11]),
                    np.array([ 1, 23, 14]),
                    np.array([ 0,  0,  0,  0, 19, 14,  2, 21,  7, 12, 11]))
    max_deg = 3
    result = gf.euclid(p1, p2, pm, max_deg=max_deg)
    assert_equal(right_answer[0], result[0])
    assert_equal(right_answer[1], result[1])
    assert_equal(right_answer[2], result[2])
