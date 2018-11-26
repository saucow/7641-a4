'''
Created on Apr 9, 2017

@author: Jon
'''
from __future__ import with_statement
import sys

sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis, getAllStates
import csv
from collections import deque


def dumpCSV(iters, times, rewards, steps, convergence, world, method, discount):
	discount = str(discount)
	discount = discount.replace(".", "_")
	fname = '%s%s-%s.csv' % (world, method, discount)
	assert len(iters) == len(times)
	assert len(iters) == len(rewards)
	assert len(iters) == len(steps)
	assert len(iters) == len(convergence)
	with open(fname, 'wb') as f:
		f.write('iter,time,reward,steps,convergence\n')
		writer = csv.writer(f, delimiter=',')
		writer.writerows(zip(iters, times, rewards, steps, convergence))


def runEvals(initialState, plan, rewardL, stepL):
	r = []
	s = []
	for trial in range(evalTrials):
		ea = plan.evaluateBehavior(initialState, rf, tf, 300);
		r.append(calcRewardInEpisode(ea))
		s.append(ea.numTimeSteps())
	rewardL.append(sum(r) / float(len(r)))
	stepL.append(sum(s) / float(len(s)))


if __name__ == '__main__':
	world = 'Easy'
	discount = 0.99
	MAX_ITERATIONS = 100;
	NUM_INTERVALS = 100;
	evalTrials = 100;
	n = 5
	userMap = [[0,0,0,0,0,0,-1,0,0,0],
	           [0,0,0,0,-3,0,0,0,0,-5],
	           [0,0,1,1,1,1,1,0,0,1],
	           [0,0,0,0,0,0,1,0,0,1],
	           [0,0,0,0,0,0,1,0,0,1],
	           [0,1,1,1,1,0,1,0,0,1],
	           [0,0,0,0,1,0,1,0,0,0],
	           [0,0,0,0,1,0,1,0,0,0],
	           [0,0,0,0,1,0,0,0,0,0],
	           [0,0,0,0,0,0,0,0,0,0]]

	# userMap = [
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	# ]

	# userMap = [
	#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
	#     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
	#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
	#     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
	#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
	#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
	#     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
	#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
	#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
	# ]

	n = len(userMap)
	# tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE,[n,n])
	# for i in range(n):
	#     for j in range(n):
	#         tmp[i][j]= userMap[i][j]
	userMap = MapPrinter().mapToMatrix(userMap)
	maxX = maxY = n - 1

	gen = BasicGridWorld(userMap, maxX, maxY)
	domain = gen.generateDomain()
	initialState = gen.getExampleState(domain);

	rf = BasicRewardFunction(maxX, maxY, userMap)
	tf = BasicTerminalFunction(maxX, maxY)
	env = SimulatedEnvironment(domain, rf, tf, initialState);
	#    Print the map that is being analyzed
	print
	"/////Easy Grid World Analysis/////\n"
	MapPrinter().printMap(MapPrinter.matrixToMap(userMap));
	visualizeInitialGridWorld(domain, gen, env)

	hashingFactory = SimpleHashableStateFactory()
	increment = MAX_ITERATIONS / NUM_INTERVALS
	timing = defaultdict(list)
	rewards = defaultdict(list)
	steps = defaultdict(list)
	convergence = defaultdict(list)
	#     # Value Iteration
	iterations = range(1, MAX_ITERATIONS + 1)

	'''
	print
	"//Easy Value Iteration Analysis//"
	
	for nIter in iterations:
		startTime = clock()
		vi = ValueIteration(domain, rf, tf, discount, hashingFactory, -1,
		                    nIter);  # //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations for comparison with the other algorithms.
		# run planning from our initial state
		vi.setDebugCode(0)
		p = vi.planFromState(initialState);
		timing['Value'].append(clock() - startTime)
		convergence['Value'].append(vi.latestDelta)
		# evaluate the policy with evalTrials roll outs
		runEvals(initialState, p, rewards['Value'], steps['Value'])
		if nIter == 1 or nIter == 25 or nIter == 50 or nIter == 15 or nIter == 100 :
			simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, 'Value Iteration %s' % (nIter))
	MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
	print
	"\n\n\n"
	# simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, 'Value Iteration %s' % (nIter))
	dumpCSV(iterations, timing['Value'], rewards['Value'], steps['Value'], convergence['Value'], world, 'Value',
	        discount=discount)
	#

	print
	"//Easy Policy Iteration Analysis//"
	for nIter in iterations:
		startTime = clock()
		pi = PolicyIteration(domain, rf, tf, discount, hashingFactory, -1, 1,
		                     nIter);  # //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations for comparison with the other algorithms.
		# run planning from our initial state
		pi.setDebugCode(0)
		p = pi.planFromState(initialState);
		timing['Policy'].append(clock() - startTime)
		convergence['Policy'].append(pi.lastPIDelta)
		# evaluate the policy with one roll out visualize the trajectory
		runEvals(initialState, p, rewards['Policy'], steps['Policy'])
		if nIter == 1 or nIter == 25 or nIter == 50 or nIter == 15 or nIter == 100:
			simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, 'Policy Iteration %s' % (nIter))
	MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
	print
	"\n\n\n"
	# simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, 'Policy Iteration %s' % (nIter))
	#     input('x')
	dumpCSV(iterations, timing['Policy'], rewards['Policy'], steps['Policy'], convergence['Policy'], world, 'Policy',
	        discount=discount)

	'''
	MAX_ITERATIONS = 10000;
	NUM_INTERVALS = 10000;
	increment = MAX_ITERATIONS / NUM_INTERVALS
	iterations = range(1, MAX_ITERATIONS + 1)
	for lr in [0.1, 0.3, 0.5, 0.9]:
		for epsilon in [0.1, 0.3, 0.5]:
			#            last10Chg = deque([99]*10,maxlen=10)
			try:
				last10Chg = deque([99] * 10, maxlen=10)
			except TypeError:
				last10Chg = deque([99] * 10)

			Qname = 'Q-Learning L%0.1f E%0.1f' % (lr, epsilon)
			agent = QLearning(domain, discount, hashingFactory, 1, lr, epsilon)
			agent.setDebugCode(0)
			print
			"//Easy {} Iteration Analysis// %s " % (Qname)
			for nIter in iterations:
				if nIter % 50 == 0: print(nIter)
				startTime = clock()
				ea = agent.runLearningEpisode(env, 300)
				if len(timing[Qname]) > 0:
					timing[Qname].append(timing[Qname][-1] + clock() - startTime)
				else:
					timing[Qname].append(clock() - startTime)
				env.resetEnvironment()
				agent.initializeForPlanning(rf, tf, 1)
				p = agent.planFromState(initialState)  # run planning from our initial state
				last10Chg.append(agent.maxQChangeInLastEpisode)
				convergence[Qname].append(sum(last10Chg) / 10.)
				# evaluate the policy with one roll out visualize the trajectory
				runEvals(initialState, p, rewards[Qname], steps[Qname])

				if nIter == 2 or nIter == 5000 or nIter == 7000 :
					Qname1 = 'Q-Learning %s L%0.1f E%0.1f' % (nIter, lr,epsilon)
					simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname1)

			MapPrinter.printPolicyMap(getAllStates(domain, rf, tf, initialState), p, gen.getMap());
			print
			"\n\n\n"
			# if lr ==0.9 and epsilon ==0.3:
			#     simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname+' {}'.format(nIter))
			#     input('s')

			# simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, 'Q-Learning L%0.1f E%0.1f' % (lr, epsilon))
			alpha = str(lr)
			alpha = alpha.replace(".", "_")
			epsilon = str(epsilon)
			epsilon = epsilon.replace(".", "_")
			temp = 'Q-Learning_L' + alpha + "_E" + epsilon
			dumpCSV(iterations, timing[Qname], rewards[Qname], steps[Qname], convergence[Qname], world, temp,
			        discount=discount)

