from inverse_iteration_with_shift import *
from rayleigh_quotient_iteration import *
from rayleigh_quotient_k_iteration import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # the user can change the following variable values as needed
    seed = 123456789
    mach_eps = 10 ** (-6)
    max_no_of_iterations = 1000
    offset = 0.001 # offset for the shift in Inverse Iteration
    k = 2 # k for RQI where the shift is updated every k-th iteration

    runtimes_ii = []
    iterations_ii = []
    res_ii = []
    rel_err_ii = []

    runtimes_rqi = []
    iterations_rqi = []
    res_rqi = []
    rel_err_rqi = []

    runtimes_rqi_k = []
    iterations_rqi_k = []
    res_rqi_k = []
    rel_err_rqi_k = []

    problem_sizes = []


    for n in range(100, 1001, 100):

        print("n = ", n)
        problem_sizes.append(n)

        # Generate a random symmetric matrix
        np.random.seed(seed)
        A = np.random.rand(n, n)
        A = (A + A.transpose()) / 2

        # Obtain the eigenvalues of matrix A and choose a random one
        eigenvalues = sp.linalg.eigvals(A)

        # Choose an eigenvalue randomly; it will be used for the shift of Inverse Iteration
        index = np.random.randint(0, len(eigenvalues))
        eigenvalue_ref_ii = eigenvalues[index]

        # Inverse Iteration Experiments

        # Set shift value based on the randomly selected eigenvalue
        # An offset is added in order to "move" the shift further from or closer to the actual eigenvalue
        sigma = eigenvalue_ref_ii + offset


        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = inverse_iteration_with_shift_function(A, sigma, seed,
                                                                                                  mach_eps,
                                                                                                  max_no_of_iterations)

        exact_eigenval = eigenvalue_ref_ii
        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)

        runtimes_ii.append(elapsed_time)
        iterations_ii.append(iterations)
        res_ii.append(res)
        rel_err_ii.append(rel_error[0][0])

        # RQI Experiments

        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = rayleigh_quotient_iteration_function(A ,seed, mach_eps, max_no_of_iterations)

        # Find the reference/exact eigenvalue for comparison purposes
        exact_eigenval = 0
        for eigenval in eigenvalues:
            if np.isclose(eigenval, approx_eigenval):
                exact_eigenval = eigenval
                break


        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)


        runtimes_rqi.append(elapsed_time)
        iterations_rqi.append(iterations)
        res_rqi.append(res)
        rel_err_rqi.append(rel_error[0][0])


        # RQI kth Iteration Experiments

        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = rayleigh_quotient_k_iteration_function(A ,seed, mach_eps, max_no_of_iterations, k)


        # Find the reference/exact eigenvalue for comparison purposes
        exact_eigenval = 0
        for eigenval in eigenvalues:
            if np.isclose(eigenval, approx_eigenval):
                exact_eigenval = eigenval
                break

        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)

        runtimes_rqi_k.append(elapsed_time)
        iterations_rqi_k.append(iterations)
        res_rqi_k.append(res)
        rel_err_rqi_k.append(rel_error[0][0])


    for n in range(2000, 5001, 1000):

        print("n = ", n)
        problem_sizes.append(n)

        # Generate a random symmetric matrix
        np.random.seed(seed)
        A = np.random.rand(n, n)
        A = (A + A.transpose()) / 2

        # Obtain the eigenvalues of matrix A and choose a random one
        eigenvalues = sp.linalg.eigvals(A)

        # Choose an eigenvalue randomly; it will be used for the shift of Inverse Iteration
        index = np.random.randint(0, len(eigenvalues))
        eigenvalue_ref_ii = eigenvalues[index]

        # Inverse Iteration Experiments

        # Set shift value based on the randomly selected eigenvalue
        # An offset is added in order to "move" the shift further from or closer to the actual eigenvalue
        sigma = eigenvalue_ref_ii + offset


        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = inverse_iteration_with_shift_function(A, sigma, seed,
                                                                                                  mach_eps,
                                                                                                  max_no_of_iterations)

        exact_eigenval = eigenvalue_ref_ii
        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)

        runtimes_ii.append(elapsed_time)
        iterations_ii.append(iterations)
        res_ii.append(res)
        rel_err_ii.append(rel_error[0][0])

        # RQI Experiments

        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = rayleigh_quotient_iteration_function(A ,seed, mach_eps, max_no_of_iterations)

        # Find the reference/exact eigenvalue for comparison purposes
        exact_eigenval = 0
        for eigenval in eigenvalues:
            if np.isclose(eigenval, approx_eigenval):
                exact_eigenval = eigenval
                break


        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)


        runtimes_rqi.append(elapsed_time)
        iterations_rqi.append(iterations)
        res_rqi.append(res)
        rel_err_rqi.append(rel_error[0][0])


        # RQI kth Iteration Experiments

        # Call the function
        approx_eigenval, x, iterations, res, elapsed_time = rayleigh_quotient_k_iteration_function(A ,seed, mach_eps, max_no_of_iterations, k)


        # Find the reference/exact eigenvalue for comparison purposes
        exact_eigenval = 0
        for eigenval in eigenvalues:
            if np.isclose(eigenval, approx_eigenval):
                exact_eigenval = eigenval
                break

        rel_error = abs(exact_eigenval - approx_eigenval) / abs(exact_eigenval)

        runtimes_rqi_k.append(elapsed_time)
        iterations_rqi_k.append(iterations)
        res_rqi_k.append(res)
        rel_err_rqi_k.append(rel_error[0][0])


    # Plotting the data

    # Runtimes
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    ii_runtime, = plt.semilogy(problem_sizes, runtimes_ii, 'r')
    rqi_runtime, = plt.semilogy(problem_sizes, runtimes_rqi, 'b')
    rqi_k_runtime, = plt.semilogy(problem_sizes, runtimes_rqi_k, 'g')
    plt.xlabel('Problem size')
    plt.ylabel('Runtime (in seconds)')
    plt.legend([ii_runtime, rqi_runtime, rqi_k_runtime], ["II", "RQI", "RQI k=2"])
    plt.savefig('runtimes.png', dpi=1200)
    plt.clf()

    # Residuals
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    ii_res, = plt.semilogy(problem_sizes, res_ii, 'r')
    rqi_res, = plt.semilogy(problem_sizes, res_rqi, 'b')
    rqi_k_res, = plt.semilogy(problem_sizes, res_rqi_k, 'g')
    plt.xlabel('Problem size')
    plt.ylabel('Residual')
    plt.legend([ii_res, rqi_res, rqi_k_res], ["II", "RQI", "RQI k=2"])
    plt.savefig('residuals.png', dpi=1200)
    plt.clf()

    #Relative errors
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    ii_rel_err, = plt.semilogy(problem_sizes, rel_err_ii, 'r')
    rqi_rel_err, = plt.semilogy(problem_sizes, rel_err_rqi, 'b')
    rqi_k_rel_err, = plt.semilogy(problem_sizes, rel_err_rqi_k, 'g')
    plt.xlabel('Problem size')
    plt.ylabel('Relative error')
    plt.legend([ii_rel_err, rqi_rel_err, rqi_k_rel_err], ["II", "RQI", "RQI k=2"])
    plt.savefig('relative_errors.png', dpi=1200)
    plt.clf()

    #Iterations
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    ii_iterations, = plt.semilogy(problem_sizes, iterations_ii, 'r')
    rqi_iterations, = plt.semilogy(problem_sizes, iterations_rqi, 'b')
    rqi_k_iterations, = plt.semilogy(problem_sizes, iterations_rqi_k, 'g')
    plt.xlabel('Problem size')
    plt.ylabel('Iterations until completion')
    plt.legend([ii_iterations, rqi_iterations, rqi_k_iterations], ["II", "RQI", "RQI k=2"])
    plt.savefig('iterations_needed.png', dpi=1200)
    plt.clf()