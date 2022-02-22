#!/usr/bin/env python3
import numpy as np
import pickle
import os
from active_sampling import LearnPolicyGradientParamsMP
from matplotlib import pyplot as plt
from mav_traj_gen import *
from planning_ros_msgs.msg import SplineTrajectory, Spline, Polynomial
from copy import deepcopy
import rospy


def run_policy(start_pos, publisher):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pg = pickle.load(open(f'{script_dir}/trained_pickles/lpgp_short_015_fast_NoVisited_10json_2000.pkl', "rb"), encoding='latin1')
    print(pg.mp_graph_file_name)
    pg.mp_graph_file_name = f'{script_dir}/trained_pickles/10.json'
    pg.load_graph()

    rewardmap = pickle.load(open(f'{script_dir}/testingData/gaussian_mixture_test2.pkl', "rb"), encoding='latin1')
    pg.orig_worldmap = np.zeros((pg.world_map_size, pg.world_map_size))
    pg.orig_worldmap[pg.pad_size:pg.pad_size+pg.reward_map_size, pg.pad_size:pg.pad_size+pg.reward_map_size] = rewardmap

    Tau = pg.generate_trajectories(1, maxPolicy=True, rand_start=False, start_pos=start_pos)
    plt.figure(figsize=(7, 6))
    plt.imshow(rewardmap, cmap='viridis', interpolation='spline36')
    for pt in Tau[0]:
        mp = deepcopy(pg.minimum_action_mp_graph[pt.index, pt.action])
        if mp is not None:
            mp.translate_start_position(pt.exact_pos - [pg.reward_map_size-1]*pg.spatial_dim)
            mp.plot(position_only=True, step_size=.01)
            plt.plot(pt.exact_pos[0]-(pg.reward_map_size-1), pt.exact_pos[1]-(pg.reward_map_size-1), 'w.')
    plt.plot(start_pos[0], start_pos[1], 'og')
    mp_list = traj_opt(Tau[0], pg, mp)
    spline_traj = mp_list_to_spline_traj(mp_list)
    publisher.publish(spline_traj)


def traj_opt(tau, pg, example_mp):
    dimension = 2

    derivative_to_optimize = derivative_order.JERK
    # derivative_to_optimize = derivative_order.SNAP
    vertices = []

    for step in tau:
        vertex = Vertex(dimension)
        vertex.addConstraint(derivative_order.POSITION, step.exact_pos - [pg.reward_map_size-1]*pg.spatial_dim)
        vertices.append(vertex)
    max_v = pg.mp_graph.max_state[1] + .5
    max_a = pg.mp_graph.max_state[2] + .5
    segment_times =estimateSegmentTimes(vertices, max_v, max_a)
    for i,seg_time in enumerate(segment_times):
        if seg_time <= 0:
            segment_times[i] = 1
   

    parameters = NonlinearOptimizationParameters()
    rho = pg.mp_graph.mp_subclass_specific_data.get('rho', 1000)
    parameters.time_penalty = rho
    opt = PolynomialOptimizationNonLinear(dimension, parameters)
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize)

    opt.addMaximumMagnitudeConstraint(derivative_order.VELOCITY, max_v)
    opt.addMaximumMagnitudeConstraint(derivative_order.ACCELERATION, max_a)

    result_code = opt.optimize()
    if result_code < 0:
        return None, None
    trajectory = Trajectory()
    opt.getTrajectory(trajectory)
    segs = trajectory.get_segments()

    mp_list = []
    for i, seg in enumerate(segs):
        traj_time = trajectory.get_segment_times()[i]
        step_size = .1
        st = np.linspace(0, traj_time, int(np.ceil(traj_time/step_size)+1))
        sp = np.zeros((dimension, st.shape[0]))
        for i, t in enumerate(st):
            sp[:, i] = seg.evaluate(t, 0)
        # plt.plot(sp[0, :], sp[1, :], 'r')
        poly_coeffs = np.array([seg.getPolynomialsRef()[i].getCoefficients(0) for i in range(dimension)])
        poly_coeffs = np.flip(poly_coeffs, axis=1)
        mp = deepcopy(example_mp)
        mp.poly_coeffs = poly_coeffs
        mp.traj_time = traj_time
        mp.start_state = mp.get_state(0)
        mp.end_state = mp.get_state(traj_time)
        mp.plot(position_only=True)
        mp_list.append(mp)
    cost = opt.getTotalCostWithSoftConstraints()
    return mp_list


def mp_list_to_spline_traj(mp_list):
    spline_traj = SplineTrajectory()
    for dimension in range(2):
        spline = Spline()
        for mp in mp_list:
            poly = Polynomial()
            poly.degree = mp.poly_coeffs.shape[1] - 1
            poly.basis = 0
            poly.dt = mp.traj_time
            # need to reverse the order due to different conventions in spline polynomial and mp.polys
            poly.coeffs = np.flip(mp.poly_coeffs[dimension, :])
            # convert between Mike's parametrization and mine
            for degree in range(len(poly.coeffs)):
                poly.coeffs[degree] *= poly.dt ** degree
            spline.segs.append(poly)
            spline.segments += 1
            spline.t_total += mp.traj_time
        spline_traj.data.append(spline)

    # add z-dim
    spline = Spline()
    for mp in mp_list:
        poly = Polynomial()
        poly.degree = mp.poly_coeffs.shape[1] - 1
        poly.basis = 0
        poly.dt = mp.traj_time
        # set poly with 0 coeffs for z direction
        poly.coeffs = np.zeros(mp.poly_coeffs.shape[1])
        # arbitrary z height 1
        poly.coeffs[0] = 1
        spline.segs.append(poly)
    spline_traj.data.append(spline)

    spline.segments = len(mp_list)
    spline.t_total = np.sum([mp.traj_time for mp in mp_list])
    spline_traj.header.frame_id = "world"
    spline_traj.header.stamp = rospy.Time.now()
    spline_traj.dimensions = 3

    return spline_traj


if __name__ == '__main__':

    rospy.init_node('lpgp_policy_publisher')

    traj_opt_pub = rospy.Publisher("/quadrotor/spline_traj", SplineTrajectory, queue_size=10)

    start_pos = np.array([2, 2])
    run_policy(start_pos, traj_opt_pub)
    # rospy.spin()
    # plt.show()
