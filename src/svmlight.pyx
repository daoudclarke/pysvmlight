# Bismillahi-r-Rahmani-r-Rahim
#
# Cython wrapper around svmlight

from libc.stdlib cimport malloc, free
from libc.string cimport strcpy

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
    strcpy(learn_parm.alphafile,"")
    strcpy(learn_parm.predfile,"trans_predictions")
    return learn_parm

cdef KERNEL_PARM get_default_kernel_parm():
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
        # List must be increasing and terminated by 0
        if 0 in python_words:
            raise ValueError("Word number must be nonzero")
        python_words = list(python_words)
        python_words.sort()
        python_words += [0]
        cdef WORD* words = <WORD*>malloc(sizeof(WORD) * len(python_words))
        cdef int i = 0
        for word in python_words:
            words[i].wnum = word
            words[i].weight = 1.0
            i += 1
        self.svector = create_svector(words, "", 1.0)

    def __repr__(self):
        values = ", ".join([
                str(self.svector.words[i].wnum)
                for i in range(self.size)])
        return "SupportVector([%s])" % values

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(len(self)):
            yield self.svector.words[i].wnum

    def __dealloc__(self):
        if self.svector is not NULL:
            free_svector(self.svector)
    
cdef class Document:
    cdef DOC _doc
    cdef object _vector

    def __cinit__(self, docnum, vector):
        self._doc.docnum = docnum
        self._doc.fvec = (<SupportVector?>vector).svector
        self._doc.queryid = 0
        self._doc.slackid = 0
        self._doc.costfactor = 1.0
        self._vector = vector

    property docnum:
        def __get__(self):
            return self._doc.docnum

    property vector:
        def __get__(self):
            return self._vector

    def __repr__(self):
        return "Document(%d, %s)" % (self._doc.docnum, self._vector.__repr__())

# cdef class DocumentList:
#     cdef DOC** _documents
    
#     def __cinit__(self, documents):
#         self._documents = <DOC**>malloc(sizeof(DOC*)*len(documents))
#         for i in range(len(documents)):
#             self._documents[i] = &(<Document?>(documents[i]))._doc
   
#     def __dealloc__(self):
#         if (self._documents):
#             free(self._documents)

class DocumentFactory:
    def __init__(self):
        self.nums = {}
        self.max_num = 0
        self.max_doc_id = 0

    def new(self, items):
        v = []
        for i in items:
            if not i in self.nums:
                self.max_num += 1
                self.nums[i] = self.max_num
            v.append(self.nums[i])
        vector = SupportVector(v)
        self.max_doc_id += 1
        return Document(self.max_doc_id, vector)
        

cdef class Model:
    # struct model:
    #     long sv_num
    #     long at_upper_bound
    #     double b
    #     DOC **supvec
    #     double *alpha
    #     long *index
    #     long totwords
    #     long totdoc
    #     KERNEL_PARM kernel_parm
    #     double loo_error,loo_recall,loo_precision
    #     double  xa_error,xa_recall,xa_precision
    #     double  *lin_weights
    #     double  maxdiff
    # ctypedef model MODEL
    cdef MODEL _model
    cdef bint _initialised

    def __cinit__(self):
        self._initialised = False
        self._model.alpha = NULL
        self._model.index = NULL
        self._model.lin_weights = NULL

    cdef void initialise(self):
        self._initialised = True

    property bias:
        def __get__(self):
            if self._initialised:
                return self._model.b
            else:
                raise ValueError("Model is invalid")

    property num_docs:
        def __get__(self):
            if self._initialised:
                return self._model.totdoc
            else:
                raise ValueError("Model is invalid")

    def __dealloc__(self):
        if self._model.alpha:
            free(self._model.alpha)
        if self._model.lin_weights:
            free(self._model.lin_weights)
        if self._model.index:
            free(self._model.index)
                 

cdef class Learner:
    cdef LEARN_PARM _parameters
    cdef KERNEL_PARM _kernel_parameters
    cdef KERNEL_CACHE _kernel_cache

    def __cinit__(self):
        self._parameters = get_default_learn_parm()
        self._kernel_parameters = get_default_kernel_parm()

    property biased_hyperplane:
        def __get__(self):
            return [False,True][self._parameters.biased_hyperplane]

        def __set__(self, value):
            self._parameters.biased_hyperplane = {False:0, True:1}[value]

    def learn(self, documents, class_values):
        cdef Model model = Model()
        cdef DOC** docs = <DOC**>malloc(sizeof(DOC*)*len(documents))
        for i in range(len(documents)):
            docs[i] = &(<Document?>(documents[i]))._doc

        cdef double* class_ = <double*>malloc(sizeof(double)*len(class_values))
        for i in range(len(class_values)):
            class_[i] = class_values[i]

        cdef long int totwords = max([max(x.vector) for x in documents])
        svm_learn_classification(docs, class_, len(documents),
                                 totwords, 
                                 &self._parameters, 
                                 &self._kernel_parameters, 
                                 &self._kernel_cache, 
                                 &model._model,
                                 NULL)
        
        free(class_)
        free(docs)
        model.initialise()
        return model

    def __repr__(self):
        return "Learner(biased_hyperplane=%s)" % str(self.biased_hyperplane)

