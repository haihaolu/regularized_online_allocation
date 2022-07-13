import numpy as np
import cvxpy as cp
import pandas as pd
import argparse
import sys
from scipy.stats import entropy
import collections

np.set_printoptions(threshold=sys.maxsize)
EPS = 1e-9

np.random.seed(123)
np.set_printoptions(precision=3)

class Dataset(object):
  def __init__(self, revenue, rho):
    self.revenue = revenue
    self.rho = rho# / np.sum(rho) * 1.5

class Problem(object):
  def __init__(self, data, regularizer, lambd):
    self.data = data
    self.lambd = lambd
    self.regularizer = regularizer
  
  # Computes argmax{x\inX} {f_t(x)-bmu^T x}
  def compute_x(self, revenue_vector, dual):
    # index = np.random.shuffle(np.argmax(revenue_vector - dual))[0]
    index = (np.argmax(revenue_vector - dual))
    x = np.zeros(len(revenue_vector))
    if revenue_vector[index] - dual[index] > 0:
      x[index] = 1
    return x, index

  # Computes argmax{x\inX} {f_t(x)-bmu^T x}
  def compute_x_with_budget(self, revenue_vector, dual, remaining_budget):
    # index = np.random.shuffle(np.argmax(revenue_vector - dual))[0]
    indexes = (np.argwhere(remaining_budget >= 1)).flatten()
    x = np.zeros(len(revenue_vector))
    if len(indexes) != 0:
      revenue_vector_shorten = revenue_vector[indexes]
      dual_shorten = dual[indexes]
      x_shorten, index_shorten = self.compute_x(revenue_vector_shorten, dual_shorten)
      index = indexes[index_shorten]
      x[index] = 1
    else:
      index = 0
    return x, index

  def compute_dual_obj(self, dual):
    # print np.max(self.data.revenue-dual, axis=1)
    T = len(self.data.revenue)
    return np.mean(np.maximum(np.max(self.data.revenue-dual, axis=1), np.zeros(T))) + (np.dot(self.data.rho, dual)+self.lambd)

  def compute_regulization(self, ave_consumption):
    if self.regularizer == "minimal_consumption":
      # print(np.min(-ave_consumption/self.data.rho))
      return np.min(ave_consumption/self.data.rho)
    if self.regularizer == "maxmin_fairness":
      return np.min(ave_consumption/self.data.rho)

def compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho, lambd):
  if not np.all(np.diff(ordered_tilde_dual*rho) >= 0):
    raise Exception("`ordered_tilde_dual*rho` must be monotonically increasing.")
  m = len(rho)
  answer = cp.Variable(m)
  objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
  constraints = []
  for i in range(1, m+1):
    constraints += [(rho[:i]*answer[:i]) >= -lambd]
  prob = cp.Problem(objective, constraints)
  prob.solve()
  return answer.value

class Algorithm(object):
  def __init__(self, reference, step_size_constant, problem):
    self.eta = step_size_constant / np.sqrt(len(problem.data.revenue))
    self.problem = problem
    self.reference = reference

  def compute_a(self, dual):
    # Computes argmax_{a<=rho} {r(a)+dual^T a}.
    if self.problem.regularizer == "minimal_consumption" or "maxmin_fairness":
      return self.problem.data.rho

  # def compute_gradient(self, dual, problem, revenue_vector):
  #   # Computes argmin_{dual' in D} V_h(dual', dual).
  #   a = self.compute_a(dual, problem)
  #   x = problem.compute_x(revenue_vector, dual)
  #   return -x + a

  def compute_next_dual(self, dual, gradient):
    # Compute the next iteration: argmin_{dual' in D} eta*gradient*dual' + V_h(dual', dual).
    if self.problem.regularizer == "minimal_consumption" and self.reference=="square_norm":
      # Solving QP: argmin_{dual' in D} eta*gradient*dual' + 1/2||dual'-dual||^2.
      answer = cp.Variable(len(dual))
      objective = cp.Minimize(self.eta * gradient.T * answer + 0.5 * cp.sum_squares(answer-dual))
      constraints = [answer >= (-self.problem.lambd/self.problem.data.rho), self.problem.data.rho.T * answer >= -self.problem.lambd]
      prob = cp.Problem(objective, constraints)
      prob.solve()
      return answer.value
    elif self.problem.regularizer == "minimal_consumption" and self.reference=="scaled_entropy":
      # dual' = ((rho.*dual+lambda)exp(-eta*gradient)-lambda)./rho
      rho = self.problem.data.rho
      lambd = self.problem.lambd
      temp = (rho*dual+lambd) * np.exp(-self.eta*gradient)
      if np.sum(temp) >= (m-1)*lambd:
        answer = (temp-lambd) / rho
      else:
        answer = ((m-1)*lambd*temp/np.sum(temp)-lambd) / rho
      return answer
    elif self.problem.regularizer == "maxmin_fairness" and self.reference == "scaled_square_norm":
      rho = self.problem.data.rho
      tilde_dual = dual - self.eta*gradient/rho/rho
      order = np.argsort(tilde_dual*rho)
      ordered_tilde_dual = tilde_dual[order]
      # print ordered_tilde_dual*rho
      ordered_next_dual = compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho[order], self.problem.lambd)
      # print(ordered_next_dual)
      # print("tilde_dual", rho*tilde_dual)
      # print("next_dual", rho*ordered_next_dual[order.argsort()])
      return ordered_next_dual[order.argsort()]
    else:
      raise Exception("The regularizer and reference function pair is not supported.")

  def mirror_descent(self, save_frequency):
    T = len(self.problem.data.revenue)
    m = len(self.problem.data.rho)
    remaining_budget = T * self.problem.data.rho
    if self.reference in ["square_norm", "scaled_square_norm"]:
      ini_dual = np.zeros(m)
    elif self.reference=="scaled_entropy":
      ini_dual = 1/np.exp(1)*np.ones(m)*self.problem.data.rho

    ind_revenue = np.zeros(m)
    ind_consumption = np.zeros(m)
    cum_revenue = np.array([0])
    cum_reg_revenue = np.array([0])
    current_dual = np.copy(ini_dual)
    sum_dual = np.copy(ini_dual)
    all_remain_budget = np.empty((0, m))
    saving_periods = np.linspace(0, T-1, num=save_frequency, dtype=int)
    all_si = np.empty((0, m))

    for t in range(T):
      revenue_vector = self.problem.data.revenue[t]
      x, index = self.problem.compute_x(revenue_vector, current_dual)
      # x, index = self.problem.compute_x_with_budget(revenue_vector, current_dual, remaining_budget)
      
      if t in saving_periods:
        # print current_dual
        all_remain_budget = np.append(all_remain_budget, np.array([remaining_budget]), axis=0)
        si = ind_consumption / self.problem.data.rho / (t+1)
        all_si = np.append(all_si, np.array([si]), axis=0)
      
      if remaining_budget[index] >= 1:
        remaining_budget[index] -= 1
        ind_revenue[index] += revenue_vector[index]
        ind_consumption[index] += 1
        cum_revenue = np.append(cum_revenue, cum_revenue[t] + revenue_vector[index])
        cum_reg_revenue = np.append(cum_reg_revenue, cum_revenue[-1] + (t+1)*self.problem.lambd*self.problem.compute_regulization(ind_consumption/(t+1)))
      else:
        cum_revenue = np.append(cum_revenue, cum_revenue[t])
        cum_reg_revenue = np.append(cum_reg_revenue, cum_revenue[-1] + (t+1)*self.problem.lambd*self.problem.compute_regulization(ind_consumption/(t+1)))

      a = self.compute_a(current_dual)
      gradient = -x + a
      current_dual = self.compute_next_dual(current_dual, gradient)
      # print(current_dual)
      sum_dual += current_dual
      # print gradient
    # print remaining_budget
    # print self.problem.compute_regulization(ind_consumption/(t+1))
    return cum_revenue[-1], cum_reg_revenue[-1], self.problem.compute_regulization(ind_consumption/(t+1)), sum_dual/(T), self.problem.compute_dual_obj(sum_dual/(T)), all_remain_budget, all_si

def load_data(instance_name, sum_rho):
  """
  Load the data.
  """
  directory = "../adx-alloc-data-2014/"
  revenue_file_name = directory + instance_name + "-sample.txt"
  revenue = np.loadtxt(revenue_file_name, delimiter=",")
  revenue = revenue / np.max(revenue)

  rho_file_name = directory + instance_name + "-ads.txt"
  rho = np.array([])
  with open(rho_file_name) as f:
    lines = f.readlines()
  for line in lines:
    rho = np.append(rho, float(line.split(":")[2][:-4]))
  return Dataset(revenue, rho/np.sum(rho)*sum_rho)
  # return Dataset(revenue, np.ones(12))

def random_data(all_data, num, size):
  """
  Create an array of num random datasets of the given size from all_data.
  """
  T = len(all_data.revenue)
  datasets = []
  for j in range(num):
    datasets.append(Dataset(all_data.revenue[np.random.randint(T, size=size)],all_data.rho))
  return datasets

def random_trials(datasets, params):
  """
  Run multiple random trials for the same dataset, and return the cumulative
  revenue for each trials.
  """
  T = len(datasets[0].revenue)
  m = len(datasets[0].rho)
  num_trials = len(datasets)
  revenue_coll = np.array([])
  real_revenue_coll = np.array([])
  dual_obj = np.array([])
  regularization_coll = np.array([])
  sum_remaining_budget = np.zeros((params.save_frequency, m))
  sum_si = np.zeros((params.save_frequency, m))
  sum_ave_dual = np.zeros(m)
  sum_ave_dual_value = 0
  for j in range(num_trials):
    data = datasets[j]
    problem = Problem(data=data, regularizer=params.regularizer, lambd=params.lambd)
    algorithm = Algorithm(reference=params.reference, step_size_constant=params.step_size_constant, problem=problem)
    real_revenue, reg_revenue, regularization, ave_dual, ave_dual_value, all_remain_budget, all_si = algorithm.mirror_descent(save_frequency=params.save_frequency)
    revenue_coll = np.append(revenue_coll, reg_revenue)
    real_revenue_coll = np.append(real_revenue_coll, real_revenue)
    regularization_coll = np.append(regularization_coll, regularization)
    sum_ave_dual += ave_dual
    sum_ave_dual_value += ave_dual_value
    sum_remaining_budget += all_remain_budget
    sum_si += all_si
  return real_revenue_coll, revenue_coll, regularization_coll, sum_ave_dual/num_trials, sum_ave_dual_value/num_trials, sum_remaining_budget/num_trials, sum_si/num_trials

if __name__ == "__main__":
  """
  Run multiple random trials of Algorithm 3 in the main paper for solving the
  proportional matching problems with high entropy, and save the regret to the
  output csv file.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_trials", help="The number of random trials.", type=int, default=1)
  parser.add_argument("--lambd", help="The coefficient of regularizer.", type=float, default=0.001)
  parser.add_argument("--data_name", help="The name of the dataset. Must be pub1-pub7.", type=str, default="pub2")
  parser.add_argument("--step_size_constant", help="The step-size constant.", type=float, default=0.001)
  parser.add_argument("--regularizer", help="The regularizer r.", type=str, default="maxmin_fairness")
  parser.add_argument("--reference", help="The reference function h.", type=str, default="scaled_square_norm")
  parser.add_argument("--save_frequency", help="How many iterations we save for the output.", type=int, default=100)
  parser.add_argument("--T_ending", help="The number of samples.", type=int, default=1000)
  parser.add_argument("--num_T", help="The number of T values.", type=int, default=2)
  parser.add_argument("--sum_rho", help="The sum of rho.", type=float, default=1.5)
  parser.add_argument("--verbosity", help="This is for testing only. `verbosity==0`: print nothing (default); `verbosity>=1`: print the output for each run; `verbosity>=2`: print the allocation decision for each iteration.", type=int, default=0)
  args = parser.parse_args()

  Params = collections.namedtuple("Params", ["lambd", "step_size_constant", "regularizer", "reference", "save_frequency", "verbosity"])
  params = Params(lambd=args.lambd,
                  step_size_constant=args.step_size_constant,
                  regularizer=args.regularizer,
                  reference=args.reference,
                  save_frequency=args.save_frequency,
                  verbosity=args.verbosity)

  if params.regularizer not in ["minimal_consumption", "maxmin_fairness"]:
    raise Exception("The regularizer must be `minimal_consumption` or `maxmin_fairness`.")

  if params.reference not in ["square_norm", "scaled_entropy", "scaled_square_norm"]:
    raise Exception("The reference must be `square_norm` or `scaled_entropy` or `scaled_square_norm`.")

  all_data = load_data(args.data_name, args.sum_rho)
  big_problem = Problem(data=all_data, regularizer=params.regularizer, lambd=params.lambd)
  m = len(all_data.rho)
  target_Ts = np.linspace(101, args.T_ending, num=args.num_T, dtype=int)

  rows = []
  columns = ["T", "Reward", "Real_Reward", "Regularization", "std", "OPT", "Regret"]

  for target_T in target_Ts:
    datasets = random_data(all_data, num=args.num_trials, size=target_T)
    real_revenue_coll, revenue_coll, regularization_coll, ave_dual, ave_dual_value, remaining_budget, ave_si = random_trials(datasets, params)
    # ave_dual_value = big_problem.compute_dual_obj(ave_dual)
    ave_revenue = np.average(revenue_coll)
    ave_real_revenue = np.average(real_revenue_coll)
    ave_regularization = np.average(regularization_coll)
    total_dual_value = target_T*(ave_dual_value)
    std = np.std(revenue_coll)
    rows.append([target_T, ave_revenue, ave_real_revenue, ave_regularization, std, total_dual_value, total_dual_value-ave_revenue])
    # print real_revenue_coll, revenue_coll, ave_dual_value, remaining_budget

  df = pd.DataFrame(rows, columns=columns)
  df.to_csv("output/results_"+args.data_name+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.regularizer)+"_"+params.reference+"_sumrho_"+str(args.sum_rho)+"_num_trials_"+str(args.num_trials)+".csv", index=False, float_format='%1.3f')
  np.savetxt("output/budget_"+args.data_name+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.regularizer)+"_"+params.reference+"_sumrho_"+str(args.sum_rho)+"_num_trials_"+str(args.num_trials)+".csv", remaining_budget, fmt='%1.3f')
  np.savetxt("output/si_"+args.data_name+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.regularizer)+"_"+params.reference+"_sumrho_"+str(args.sum_rho)+"_num_trials_"+str(args.num_trials)+".csv", ave_si, fmt='%1.3f')
  np.savetxt("output/real_revenue_"+args.data_name+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.regularizer)+"_"+params.reference+"_sumrho_"+str(args.sum_rho)+"_num_trials_"+str(args.num_trials)+".csv", real_revenue_coll, fmt='%1.3f')
  np.savetxt("output/regularization_"+args.data_name+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.regularizer)+"_"+params.reference+"_sumrho_"+str(args.sum_rho)+"_num_trials_"+str(args.num_trials)+".csv", regularization_coll, fmt='%1.3f')

  # m = len(dataset.rho)
  # T = len(dataset.revenue)
  # problem = Problem(data=dataset, regularizer="minimal_consumption", lambd=0.01)
  # algorithm = Algorithm(reference="square_norm", step_size_constant=0.1, problem=problem)
  # cum_reg_revenue, ave_dual, all_remain_budget = algorithm.mirror_descent(save_frequency=10)
  # # ave_dual = np.ones(m)
  # dual_obj = T*problem.compute_dual_obj(ave_dual)
  # print(cum_reg_revenue)
  # print(len(all_remain_budget))


  """
  Run multiple random trials of Algorithm 3 in the main paper for solving the
  proportional matching problems with high entropy, and save the regret to the
  output csv file.
  """
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--num_trials", help="The number of outer random loop.", type=int, default=1)
  # parser.add_argument("--lambd", help="The coefficient of regularizer.", type=float, default=0.1)
  # parser.add_argument("--data_name", help="The name of the dataset. Must be pub1-pub7.", type=str, default="pub2")
  # parser.add_argument("--step_size_constant", help="The step-size of the algorithm is defined by step_size_constant/sqrt{T}, where T is number of online stEPS. The default value (0) corresponds to step_size_constant=1 for gd, and step_size_constant=10 for ew.", type=float, default=0.0)
  # parser.add_argument("--verbosity", help="This is for testing only. `verbosity==0`: print nothing (default); `verbosity>=1`: print the output for each run; `verbosity>=2`: print the allocation decision for each iteration.", type=int, default=0)
  # args = parser.parse_args()

  # Params = collections.namedtuple("Params", ["num_trials", "lambd", "step_size_constant", "method", "termination", "restricted_choice", "verbosity", "smooth_delivery", "smooth_delivery_constant"])
  # params = Params(num_first=args.num_first,
  #                 num_sec=args.num_sec,
  #                 lambd=args.lambd,
  #                 step_size_constant=args.step_size_constant,
  #                 method=args.method,
  #                 termination=args.termination,
  #                 restricted_choice=args.restricted_choice,
  #                 smooth_delivery=args.smooth_delivery,
  #                 smooth_delivery_constant=args.smooth_delivery_constant,
  #                 verbosity=args.verbosity)

  # if params.method not in ["gd", "ew", "mr"]:
  #   raise Exception("The method needs to be gd or ew.")

  # if params.smooth_delivery not in ["off", "exp", "poly", "linear"]:
  #   raise Exception("The smooth delivery option needs to be `off` or `exp` or `poly` or `linear`.")

  # if params.termination != "stopping_time" and params.termination !=  "exhaustion":
  #   raise Exception("The termination needs to be stopping_time or exhaustion.")

  # all_data = load_data(args.data_name)
  # m = len(all_data.rho)
  # T = len(all_data.revenue)
  # target_Ts = np.linspace(100, 10000, num=16, dtype=int)
  # # optimal_value = compute_dual_value(all_data, lambd)
  # # print optimal_value
  
  # rows = []
  # columns = ["T", "Reward", "Real_Reward", "std", "OPT", "Regret"]
  # for target_T in target_Ts:
  #   datasets = random_data(all_data, num=params.num_first, size=target_T)
  #   T = len(datasets[0].revenue)
  #   if params.step_size_constant == 0:
  #     if params.method == "gd":
  #       eta = 1 / np.sqrt(T)
  #     elif params.method == "ew":
  #       eta = 10 / np.sqrt(T)
  #   else:
  #     eta = params.step_size_constant / np.sqrt(T)
  #   revenue_coll, real_revenue_coll, dual_coll  = first_randomness(datasets, params)
  #   ave_revenue = np.average(revenue_coll)
  #   ave_real_revenue = np.average(real_revenue_coll)
  #   ave_dual_value = T*np.average(dual_coll)
  #   std = np.std(revenue_coll)
  #   rows.append([T, ave_revenue, ave_real_revenue, std, ave_dual_value, ave_dual_value-ave_revenue])
  # df = pd.DataFrame(rows, columns=columns)
  # df.to_csv("results"+str(args.num_first)+"_"+str(args.num_sec)+"_"+args.termination+"_"+args.data_name+"_"+args.method+"_lambda"+str(params.lambd).replace('.', "")+"_ss"+str(params.step_size_constant).replace('.', "")+"_"+str(args.restricted_choice)+"_"+params.smooth_delivery+".csv", index=False)


