# Bismillahi-r-Rahmani-r-Rahim
#
# Cython wrapper around svmlight

from libc.stdlib cimport malloc, free

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

    # Functions

    cdef SVECTOR *create_svector(WORD *, char *, double)
    cdef void free_svector(SVECTOR *)


cdef extern from "svm_learn.h":
    void svm_learn_classification(DOC **docs, double *class_, long int
			      totdoc, long int totwords, 
			      LEARN_PARM *learn_parm, 
			      KERNEL_PARM *kernel_parm, 
			      KERNEL_CACHE *kernel_cache, 
			      MODEL *model,
			      double *alpha)

cdef LEARN_PARM get_default_learn_parm():
    cdef LEARN_PARM learn_parm
    learn_parm.biased_hyperplane=1
    learn_parm.sharedslack=0
    learn_parm.remove_inconsistent=0
    learn_parm.skip_final_opt_check=0
    learn_parm.svm_maxqpsize=10
    learn_parm.svm_newvarsinqp=0
    learn_parm.svm_iter_to_shrink=-9999
    learn_parm.maxiter=100000
    learn_parm.kernel_cache_size=40
    learn_parm.svm_c=0.0
    learn_parm.eps=0.1
    learn_parm.transduction_posratio=-1.0
    learn_parm.svm_costratio=1.0
    learn_parm.svm_costratio_unlab=1.0
    learn_parm.svm_unlabbound=1E-5
    learn_parm.epsilon_crit=0.001
    learn_parm.epsilon_a=1E-15
    learn_parm.compute_loo=0
    learn_parm.rho=1.0
    learn_parm.xa_depth=0
    return learn_parm

cdef get_default_kernel_parm():
    cdef KERNEL_PARM parm
    parm.kernel_type=0
    parm.poly_degree=3
    parm.rbf_gamma=1.0
    parm.coef_lin=1
    parm.coef_const=1
    return parm

def test_range(r):
    cdef int ii
    for i in r:
        ii = i
        print ii

cdef class SupportVector:
    cdef SVECTOR* svector
    cdef public int size

    def __cinit__(self, python_words):
        self.size = len(python_words)
        cdef WORD* words = <WORD*>malloc(sizeof(WORD) * self.size)
        cdef int i = 0
        for word in python_words:
            words[i].wnum = word
            words[i].weight = 1.0
            i += 1
        self.svector = create_svector(words, "", 1.0)

    def __repr__(self):
        values = ",".join([
                str(self.svector.words[i].wnum)
                for i in range(self.size)])
        return "SupportVector([%s])" % values

    def __len__(self):
        return self.size

    def __dealloc__(self):
        if self.svector is not NULL:
            free_svector(self.svector)
    

