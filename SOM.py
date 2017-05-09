import pandas as pd
import tensorflow as tf
import tensorflow as tf
import numpy as np
import os

class SOM(object):
    """
    N-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m ,dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(*m).
        """

        # Assign required variables first
        self._m = m

        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = np.max(self._m) /  2

        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m1*m2.., dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [np.prod(self._m), dim]))

            # Matrix for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(self._neuron_locations(self._m)))
            ##PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vector
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(np.prod(self._m))])), 2), 1)),
                0)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, len(self._m)]))),
                                 [len(self._m)])

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(tf.cast(alpha , "float32"), learning_rate_op)
            _sigma_op = tf.multiply(tf.cast(sigma , "float32"), learning_rate_op)

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(np.prod(self._m))])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                                for i in range(np.prod(self._m))])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(np.prod(self._m))]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _neuron_locations(self, m):
        """
        Yields one by one the N-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all N-D locations in the map
        grid = [np.arange(x) for x in m]

        mesh = np.meshgrid(*grid)
        w = [x.flatten() for x in mesh]
        z = np.vstack(tuple(w)).T
        return z

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        with self._graph.as_default():
            saver = tf.train.Saver()
            if os.path.exists("checkpoint") != True:
            # Training iterations
                for iter_no in range(self._n_iterations):
                    # Train with each vector one by one
                    for input_vect in input_vects:
                        self._sess.run(self._training_op,
                                       feed_dict={self._vect_input: input_vect,
                                                  self._iter_input: iter_no})

                # Store a centroid grid for easy retrieval later on
                centroid_grid = {}
                self._weightages = list(self._sess.run(self._weightage_vects))
                self._locations = list(self._sess.run(self._location_vects))
                for i, loc in enumerate(self._locations):
                    centroid_grid[str(loc)] = self._weightages[i]
                self._centroid_grid = centroid_grid

                saver.save(self._sess, "./")
            else:
                saver.restore(self._sess, "./")
                print("Restored")
                centroid_grid = {}
                self._weightages = list(self._sess.run(self._weightage_vects))
                self._locations = list(self._sess.run(self._location_vects))
                for i, loc in enumerate(self._locations):
                    centroid_grid[str(loc)] = self._weightages[i]
                self._centroid_grid = centroid_grid

            self._trained = True

    def get_centroids(self):
        """
        Returns a dictionary of centroids with key as location and values as weights
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return