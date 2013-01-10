
/**
 * Elastic net regulation for linear regression using IGD optimizer
 */

/**
 * @brief Linear regression (incremental gradient): Transition function
 */
DECLARE_UDF(convex, gaussian_igd_transition)

/**
 * @brief Linear regression (incremental gradient): State merge function
 */
DECLARE_UDF(convex, gaussian_igd_merge)

/**
 * @brief Linear regression (incremental gradient): Final function
 */
DECLARE_UDF(convex, gaussian_igd_final)

/**
 * @brief Linear regression (incremental gradient): Difference in
 *     log-likelihood between two transition states
 */
DECLARE_UDF(convex, internal_gaussian_igd_distance)

/**
 * @brief Linear regression (incremental gradient): Convert
 *     transition state to result tuple
 */
DECLARE_UDF(convex, internal_gaussian_igd_result)
