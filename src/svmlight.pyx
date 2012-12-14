# Bismillahi-r-Rahmani-r-Rahim
#
# Cython wrapper around svmlight

cdef extern from "svm_common.h":
    struct word:
        long wnum
        float weight
    ctypedef word WORD

    ctypedef svector SVECTOR
    struct svector:
        WORD *words
        double twonorm_sq
        char *userdefined
        long kernel_id
        SVECTOR *next
        double  factor

    
    struct doc:
        long docnum
        long queryid
        double costfactor
        long slackid
        SVECTOR *fvec
    ctypedef doc DOC

    struct kernel_parm:
        long kernel_type
        long poly_degree
        double rbf_gamma
        double coef_lin
        double coef_const
        char custom[50]
    ctypedef kernel_parm KERNEL_PARM

    struct model:
        long sv_num
        long at_upper_bound
        double b
        DOC **supvec
        double *alpha
        long *index
        long totwords
        long totdoc
        KERNEL_PARM kernel_parm
        double loo_error,loo_recall,loo_precision
        double  xa_error,xa_recall,xa_precision
        double  *lin_weights
        double  maxdiff
    ctypedef model MODEL

    struct kernel_cache:
        long *index
        float *buffer
        long   *invindex
        long   *active2totdoc
        long   *totdoc2active
        long   *lru
        long   *occu
        long   elems
        long   max_elems
        long   time
        long   activenum
        long   buffsize
    ctypedef kernel_cache KERNEL_CACHE

    struct learn_parm:
        long   type   # selects between regression and classification
        double svm_c  # upper bound C on alphas
        double eps    # regression epsilon (eps=1.0 for classification
        double svm_costratio         # factor to multiply C for positive examples
        double transduction_posratio # fraction of unlabeled examples to be classified as positives
        long   biased_hyperplane     # if nonzero, use hyperplane w*x+b=0 otherwise w*x=0
        long   sharedslack           # if nonzero, it will use the shared
                                     # slack variable mode in
                                     # svm_learn_optimization. It requires
                                     # that the slackid is set for every
                                     # training example
        long   svm_maxqpsize         # size q of working set
        long   svm_newvarsinqp       # new variables to enter the working set in each iteration
        long   kernel_cache_size     # size of kernel cache in megabytes
        double epsilon_crit          # tolerable error for distances used in stopping criterion
        double epsilon_shrink        # how much a multiplier should be above zero for shrinking
        long   svm_iter_to_shrink    # iterations h after which an example can be removed by shrinking
        long   maxiter               # number of iterations after which the
				     # optimizer terminates, if there was
				     # no progress in maxdiff */
        long   remove_inconsistent   # exclude examples with alpha at C and retrain
        long   skip_final_opt_check  # do not check KT-Conditions at the end of
				     # optimization for examples removed by 
				     # shrinking. WARNING: This might lead to 
				     # sub-optimal solutions! */
        long   compute_loo           # if nonzero, computes leave-one-out estimates
        double rho                   # parameter in xi/alpha-estimates and for
				     # pruning leave-one-out range [1..2] */
        long   xa_depth              # parameter in xi/alpha-estimates upper
                                     # bounding the number of SV the current
				     # alpha_t is distributed over */
        char predfile[200]           # file for predicitions on unlabeled examples in transduction */
        char alphafile[200]          # file to store optimal alphas in. use  
				     # empty string if alphas should not be output */

        #/* you probably do not want to touch the following */
        double epsilon_const         # tolerable error on eq-constraint */
        double epsilon_a             # tolerable error on alphas at bounds */
        double opt_precision         # precision of solver, set to e.g. 1e-21 if you get convergence problems */

        #/* the following are only for internal use */
        long   svm_c_steps           # do so many steps for finding optimal C */
        double svm_c_factor          # increase C by this factor every step */
        double svm_costratio_unlab
        double svm_unlabbound
        double *svm_cost             # individual upper bounds for each var */
        long   totwords              # number of features */
    ctypedef learn_parm LEARN_PARM



cdef extern from "svm_learn.h":
    void svm_learn_classification(DOC **docs, double *class_, long int
			      totdoc, long int totwords, 
			      LEARN_PARM *learn_parm, 
			      KERNEL_PARM *kernel_parm, 
			      KERNEL_CACHE *kernel_cache, 
			      MODEL *model,
			      double *alpha)


