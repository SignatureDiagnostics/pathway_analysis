import numpy as np
from numpy.random import default_rng
rng = np.random.default_rng(seed=82454) # to make random sampling replicable
from itertools import combinations, product
import warnings
import copy
from functools import wraps
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})
plt.ioff()


class CTBase:
    ''' Builds a counts table for a single group of sample ids and postprocesses the samples. 
    
    Parameters
    ----------
    
    sample_ids: list
        A list of sample ids.  Note that these lists can be created from the attributes of patient
        records (e.g., Black women who deliver before week 37).  They don't have to be predefined
        groups such as 'preterm birth'.
        
    group_label: str
        A string that identifies the group
        
    from_table: pandas dataframe
        Sometimes we want to create a counts table from a dataframe of counts
        
    regions: list of str
        To access pattern_probs, which is organized as {sample_id : {region : {pattern: prob}}}, 
        regions are identified by strings such as chromosome names or gene promoter region names.
    
    patterns: list of str
        These are methylation patterns such as '00'. 
    
    n_ss: int
        The number of subsamples to draw for each sample in sample_ids.
        
    postprocess: function or lambda expression
        This function takes a df of counts, `ct`, and produces a df of processed counts, `ctp`.
    
    kwds: 
        parameters that may be passed to postprocess_fn
    
    
    Attributes
    ----------
    
    ct: pd.DataFrame
        df of counts
    
    ctp: pd.DataFrame
        df of postprocessed counts
    
    ctpair: pd.DataFrame
        df of exactly K columns from ct or ctp, used to build a PIRM of order K
    
    '''
    
    def __init__(self,sample_ids=None, group_label=None, postprocess = None, **kwds):
        self.unique_ids = sample_ids
        self.group_label = group_label
        self.postprocess = postprocess
        
    def make_ct (self):
        pass
    
    def subsample (self,sample_id,n_subsamples=None,N=None):
        ''' Draws a batch of subsamples for a single sample.  The sample_id must be in the 
        sample ids in this CT object.  To subsample _any_ sample id, use the SubSample class.'''
        pass
    
    def hold_out (self,sample_id):
        ''' Returns all ctp records except those with the sample_id. '''
        return self.ctp[self.ids != sample_id]
    
    
class CT (CTBase):

    def __init__(self,pattern_probs,regions=None, patterns=None, regions_patterns = None,
                 n_subsamples = 30, N = 30000, **kwds):   
        self.pattern_probs = pattern_probs
        self.n_ss = n_subsamples
        self.N = N
        super().__init__(**kwds)
        
        # takes region_pattern lists (e.g., [['chr9','010'],['chr1','11']]) if specified otherwise
        # creates region_pattern list as all pairwise combinations of regions and patterns
        
        self.regions_patterns = regions_patterns or list(product(regions,patterns))
        
        # make the counts table
        self.ct, self.ids = self.make_ct()
        
        # postprocess the counts table
        self.ctp = self.postprocess(self.ct,**kwds) if self.postprocess is not None else self.ct
    
    def make_ct (self):
        _ids,_counts = [],[]

        # Get `n_subsamples` subsamples for each sample.  Each subsample is of size N.
        for _id in self.unique_ids:
            
            # get pattern probs for this _id
            probs = [self.pattern_probs[_id][region][pattern] for region,pattern in self.regions_patterns]
            
            # get given number of subsamples for this _id for these probs
            _counts.extend(np.array([rng.binomial(self.N,p,self.n_ss) for p in probs]).T)
            _ids.extend([_id]*self.n_ss)
            
        return pd.DataFrame(_counts,columns=self.regions_patterns), np.array(_ids)
    
    def subsample (self,sample_id,n_subsamples=None,N=None):
        ''' Draws a batch of subsamples for a single sample.  The sample_id must be in the 
        sample ids in this CT object.  To subsample _any_ sample id, use the SubSample class.'''
        
        return SubSample(
            pattern_probs = self.pattern_probs,
            sample_ids = sample_id, 
            group_label = self.group_label,
            regions_patterns = self.regions_patterns,
            n_subsamples = n_subsamples or self.n_ss,
            postprocess = self.postprocess, 
            N = N or self.N)

class SubSample (CT):
    ''' SubSamples is a subclass of CT for which sample_ids is a single sample (e.g., 'PL1927') '''
    
    def make_ct (self):
        # Get `n_subsamples` subsamples for each sample.  Each subsample is of size N.
        _counts = []
        probs = [self.pattern_probs[self.unique_ids][region][pattern] for region,pattern in self.regions_patterns]
        _counts = np.array([rng.binomial(self.N,p,self.n_ss) for p in probs]).T
        _ids = [self.unique_ids]*self.n_ss
        return pd.DataFrame(_counts,columns=self.regions_patterns), _ids
    
    
class CT_From_Table (CTBase):
    
    def __init__(self,table=None,**kwds):
        super().__init__(**kwds)
        
        # make the counts table
        self.ct, self.ids = table, table.index
        
        # by default there is no subsampling
        self.n_ss = 1
        
        # postprocess the counts table
        self.ctp = self.postprocess(self.ct,**kwds) if self.postprocess is not None else self.ct
    
    
    def subsample (self,sample_id,n_subsamples=None,N=None):
        ''' Currently there is no subsampling method for counts tables that are
        identical with tabular data such as FPKM tables. A call to subsample 
        thus returns self.ctp.'''
        return self.ctp
 
def hypo_hyper_mixed (table,**kwds):
    '''
    reduce the table to have just hypo, hyper and mixed columns for each region
     e.g., sum frequencies in '1' and '11' into a single 'hyper' column
     The resulting numbers are not probabiliites, but probabilities out of N
     So for example if your N is 1000, then the resulting number is out of N
     fragments, how many would by hyper or hypo methylated
    '''
    cols = table.columns
    regions = set([x[0] for x in cols])

    hypo_cols =  [[c for c in cols if '1' not in c[1] and c[0] == _chr] for _chr in regions]
    hyper_cols = [[c for c in cols if '0' not in c[1] and c[0] == _chr] for _chr in regions]
    mixed_cols = [[c for c in cols if '0' in c[1] and '1' in c[1] and c[0] == _chr] for _chr in regions]
    acc = []
    new_cols = []

    for bundles, label in zip([hypo_cols,hyper_cols,mixed_cols],['hypo','hyper','mixed']):
        for bundle in bundles:
            if bundle != []:
                acc.append(list(table[bundle].sum(axis=1)))
                new_cols.append(f'{bundle[0][0]}_{label}')

    return pd.DataFrame(np.array(acc).T ,columns=new_cols)


def standardize (table, means= None, stds = None):
    
    if means is None and stds is None:
        # we have to calculate them, otherwise we'll use the given means,stds
        means = table.mean(axis=0)
        stds  = table.std(axis=0)
        
    Z = (table - means)/stds
    return Z, means, stds,


def reduce_and_standardize (table,means=None,stds=None):
    return standardize(hypo_hyper_mixed (table), means, stds)

class Screen ():
    ''' Given a table that contains both cases and controls, and a postprocessing pipeline,
    and a pair of regions with which to build PIRMs, this builds a screen that can process new 
    data and classify it as case or control. If ho_case or ho_ctrl is specified, this holds out 
    the given sample ids when building PIRMs.'''
    
    def rescale (self,x):
        ''' x ranges from 0..1, this rescales x to -1..1 '''
        return 2 * x - 1
    
    def __init__(self,name, ct_case, ct_ctrl, pair):
        
        self.name=name
        self.pair = pair
        self.label_case = ct_case.group_label
        self.label_ctrl = ct_ctrl.group_label

        self.xy_case = ct_case.ctp[self.pair].to_numpy()
        self.xy_ctrl = ct_ctrl.ctp[self.pair].to_numpy()

        self.m_case, self.b_case = np.polyfit(self.xy_case[:,0],self.xy_case[:,1],1)
        self.m_ctrl, self.b_ctrl = np.polyfit(self.xy_ctrl[:,0],self.xy_ctrl[:,1],1)   

    def classify (self,sample,sample_id=None,sample_label=None,rescale=False):
        ''' Takes a CT or SubSample object or a ctp for a single sample.  Extracts the columns designated
        by self.pair, the first treated as X, the second as Y. The sample_id and sample true label are optional.  
        Returns a dict with a classification of the sample and some metrics.'''
        
        if hasattr(sample,'ctp'):
            ctp = sample.ctp # sample is a CT or SubSample object
        elif type(sample) == pd.DataFrame:
            ctp = sample # sample is probably a ctp
        else:
            raise ValueError('sample must be a CT object, a SubSample object, or the ctp from one of those')
        
        # We only want the screen pair, irrespective of other columns in ctp
        ctp = ctp[self.pair].to_numpy()
        
        err_case = (ctp[:,1] - (self.m_case * ctp[:,0] + self.b_case))**2
        err_ctrl = (ctp[:,1] - (self.m_ctrl * ctp[:,0] + self.b_ctrl))**2
        n = len(ctp)
        case_wins = np.sum(err_case < err_ctrl)
        ctrl_wins = n - case_wins
        
        pred = self.label_case if case_wins > ctrl_wins else self.label_ctrl

        err = err_case if case_wins > ctrl_wins else err_ctrl

        
        return {'screen': self.name, 'sample' : sample_id, 'true': sample_label, 'pred' : pred, 'err' : err} 
                


def textCM (tp,fn,tn,fp):
    ''' Prints a 2x2 confusion matrix given tp, fn, tn, fp '''
    print(f"\n{'true':^50}\n{'case':>22}{'ctrl':>10}\n\t\t---------+--------")
    print(f"{'case':>15}{tp:^10}{'|'}{fn:^10}")
    print(f"pred\t\t---------|--------")
    print(f"{'ctrl':>15}{fp:^10}{'|'}{tn:^10}")
    print(f"\t\t---------+--------\n")
    

def case_ctrl_loo (ct_case,ct_ctrl,pair,stats = ['sens','spec','tp','fn','tn','fp']):
    ''' 
    Leave-one-out test for case/ctrl given a PIRM pair. Returns statistics in the
    order specified by the stats parameter.
    
    Each sample in case is held out, a case model is fit, and then each held-out
    case sample is classified based on the held-out model vs. the ctrl model fit
    to all ctrl samples.  Then each sample in ctrl is held out, a ctrl model is 
    fit, and each held out ctrl sample is classified based on the held-out model 
    and the case model. 
    '''
    
    def loo_group (ct, other_params):
        # hit means the ct model is closer than the other model, miss means not hit
        hit,miss = 0,0
        # params of the non-ct model (e.g., if ct is case, other_params are from ctrl model)
        other_m,other_b = other_params
        
        xy = ct.ctp[pair].to_numpy()
        
        # leave-one-out modeling and classification
        for _id in ct.unique_ids:
            # get training and testing sets
            select = ct.ids == _id
            train = xy[~select]
            test = xy[select]
            
            # train model on case training samples
            m,b = np.polyfit(train[:,0],train[:,1],1)

            # test the held out sample with both case and control models
            SSres = (test[:,1] - (m * test[:,0] + b))**2
            SSres_other = (test[:,1] - (other_m * test[:,0] + other_b))**2

            # if more than half of the replicates (if any) have lower residuals
            # from the ct line than from the other line, then ct wins
            ct_wins = np.sum(SSres < SSres_other)
            if ct_wins > len(SSres)/2:
                hit += 1
            else:
                miss += 1
        return hit,miss
    
    xy_case = ct_case.ctp[pair].to_numpy()
    xy_ctrl = ct_ctrl.ctp[pair].to_numpy()
    case_params = np.polyfit(xy_case[:,0],xy_case[:,1],1) 
    ctrl_params = np.polyfit(xy_ctrl[:,0],xy_ctrl[:,1],1)
    
    tp,fn = loo_group(ct_case, ctrl_params)
    tn,fp = loo_group(ct_ctrl, case_params)
    
    # In case TP+FP = 0 or TN+FN = 0, the denominator will be zero.
    # In such cases, in the limit mcc = 0
    
    try:
        mcc = ((tn*tp)-(fp*fn))/(((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp))**.5)
    except:
        #print("Warning:  Denominator of mcc == 0")
        mcc = 0
    
    f1 = (2*tp)/(2*tp+fn+fp)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    
    stats_dict = {'tp':tp,'fn':fn,'tn':tn,'fp':fp,'mcc':mcc,'f1':f1,'sens':sens,'spec':spec}
    
    return [stats_dict[stat] for stat in stats]

class DiscoverySearch ():    
    def __init__(self,ct_case,ct_ctrl):
        self.ct_case = ct_case
        self.ct_ctrl = ct_ctrl
        
        # make sure that case and ctrl counts tables have the same columns
        assert all(self.ct_case.ctp.columns == self.ct_ctrl.ctp.columns)
        
        self.columns = np.array(self.ct_case.ctp.columns)
        
        # the number of subsamples in each of ct_case and ct_ctrl, respectively
        self.case_n_ss = self.ct_case.n_ss
        self.ctrl_n_ss = self.ct_ctrl.n_ss
        
        # the number of unique cases and controls
        self.n_cases = int(self.ct_case.ctp.shape[0]/self.case_n_ss)
        self.n_ctrls = int(self.ct_ctrl.ctp.shape[0]/self.ctrl_n_ss)
        self.n_cols = len(self.columns)

    def test_all_combos (self,np_case=None,np_ctrl=None,threshold_case=.5, threshold_control=.5, find_no_call = False):
        '''
        If np_case and np_ctrl are not None then we are passing self.ct_case.ctp.to_numpy() and
        self.ct_ctrl.ctp.to_numpy().  Otherwise we compute these Numpy arrays inside this method.
        We will pass np_case and np_ctrl when these are resampled pseudosamples. 
        
        For N columns in counts tables for cases and controls, this calculates the regressions 
        of N-1 columns against column 0 for each of case and control.  It does this N times, 
        rolling the columns by 1 position after each iteration.  The result is a dict with N keys, 
        each of which is a column 0. The values for each column 0 are the results of a classifier 
        based on the regressions of all other columns on column 0, specifically, mcc, f1, sens, 
        spec, tp, fn, tn, fp.  
        
        E.g., {'chr11_hypo': {'chr21_hypo': [0.8640987,0.8888888,1.0,0.93333333,12,0,42,3]} ... }
        is the results of a classifier based on regressing chr21_hypo on chr11_hypo.
        
        This works for any number of subsamples per sample, including just one subsample.
        '''
        
        #if you are not finding nocall, then the threshold must be 5 otherwise there is data with no way to classify
        if find_no_call == False and (threshold_case != .5 or threshold_control !=.5):
            print('THRESHOLD AUTOMATICALLY RESET TO .5')
            threshold_case = .5
            threshold_control = .5
        
        case = self.ct_case.ctp.to_numpy() if np_case is None else np_case
        ctrl = self.ct_ctrl.ctp.to_numpy() if np_ctrl is None else np_ctrl
        cols = copy.copy(self.columns)
        acc= []

        for _ in range(self.n_cols):

            # This calculates m and b for the regression of each column after the first
            # on the first column.  m_case, b_case etc are vectors of N-1 parameters where
            # N is the number of columns

            X_case,Y_case = case[:,0],case[:,1:]
            X_ctrl,Y_ctrl = ctrl[:,0],ctrl[:,1:]

            try:
                m_case,b_case = np.polyfit(X_case,Y_case,1)
            except:
                print('test_all_combos broke in polyfit case')
                break
            try:
                m_ctrl,b_ctrl = np.polyfit(X_ctrl,Y_ctrl,1)
            except:
                print('test_all_combos broke in polyfit ctrl')
                break

            # Now calculate sum of square deviations: case_case means SS for case data
            # predicted by case model; case_ctrl is SS for case data predicted by ctrl model
            X_case, X_ctrl =  X_case[:,None], X_ctrl[:,None]

            # squared deviations over pseudosamples           
            case_case = (Y_case - (m_case * X_case + b_case))**2
            case_ctrl = (Y_case - (m_ctrl * X_case + b_ctrl))**2
            ctrl_ctrl = (Y_ctrl - (m_ctrl * X_ctrl + b_ctrl))**2
            ctrl_case = (Y_ctrl - (m_case * X_ctrl + b_case))**2
            
            '''
            case_wins says whether a case is closer to the case PIRM than to the ctrl PIRM. 
            The dimensions of case_wins are the number of case pseudosamples by the number 
            of regressions against the 0th column. case_wins must be turned into true positives
            and false negatives.  First we reshape case_wins into three dimensions with shape
            [n_samples, n_subsamples per sample, n_regressions].  Then we get the mean number
            of subsamples classified correctly.  If this exceeds a theshold we say the sample 
            is classified correctly.
            '''

            case_wins = (case_ctrl > case_case)
            # Reshape into a 3D table where axis 0 is samples, axis 1 is subsamples for each sample
            # and axis 2 is regressions.  This is a boolean table in which True means the subsample is
            # closer to the case regression line than the control regression line.
            case_wins_reshaped = case_wins.reshape(self.n_cases,self.case_n_ss,self.n_cols-1)
            # The mean of the boolean table across subsamples (axis 1) will be a probability that a
            # case sample (axis 0) is closer to a case regression (axis 2). case_wins_prob is a 2D
            # array of floats  where axis 0 is samples and axis 1 is regressions
           
            case_wins_probs = case_wins_reshaped.mean(axis=1)
            # clear_case_wins are samples in which case_wins_probs exceeds a threshold
            
            clear_case_wins = case_wins_probs > threshold_case

            # clear_case_losses are samples in which case_wins_probs are lower than 1-threshold
            clear_case_losses = case_wins_probs <= 1-threshold_case

            #case_uncertain are samples in which the probabilities are in between the threshold and 1- threshold
            case_uncertain = ~(clear_case_wins | clear_case_losses)
            
            # Summing clear case wins over samples gives the number of clear case wins for each regression
            # case_regression_scores is a vector in which each cell represents the number of cases classified
            # as cases by a regression.
            
            cases_classified_as_cases = clear_case_wins.sum(axis=0)
            
            # Do it again for ctrl
            ctrl_wins = (ctrl_case > ctrl_ctrl)
            ctrl_wins_reshaped = ctrl_wins.reshape(self.n_ctrls,self.ctrl_n_ss,self.n_cols-1)
            ctrl_wins_probs = ctrl_wins_reshaped.mean(axis=1)
            clear_ctrl_wins = ctrl_wins_probs > threshold_control
            clear_ctrl_losses = ctrl_wins_probs <= 1-threshold_control

            ctrl_uncertain = ~(clear_ctrl_wins | clear_ctrl_losses)

            ctrls_classified_as_ctrls = clear_ctrl_wins.sum(axis=0)

            # Accuracy measures:  tp, fn, tn, fp, Matthews correlation coefficient, f1 score,
            # sens and spec. Calculate these for all regressions simultaneously.
            
            tp = cases_classified_as_cases
            fn = (self.n_cases - cases_classified_as_cases)
            tn = ctrls_classified_as_ctrls
            fp = (self.n_ctrls - ctrls_classified_as_ctrls)

            # In case tp+fp = 0 or tn+fn = 0, the denominator will be zero.
            # In such cases, in the limit mcc = 0
    
            if ((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp)).any():
                mcc = ((tn*tp)-(fp*fn))/(((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp))**.5)
            else:
                mcc = 0
            f1 = (2*tp)/(2*tp+fn+fp)
            sens = tp/(tp+fn)
            spec = tn/(tn+fp)
            no_call = (case_uncertain.sum(axis=0) + ctrl_uncertain.sum(axis=0))/(tp+tn+fn+fp+case_uncertain.sum(axis=0) + ctrl_uncertain.sum(axis=0))

            if find_no_call:
                results = zip(cols[1:],mcc,f1,sens,spec,tp,fn,tn,fp,no_call, 
                            clear_ctrl_wins.transpose(),
                            clear_ctrl_losses.transpose(), 
                            ctrl_uncertain.transpose(),
                            clear_case_wins.transpose(),
                            clear_case_losses.transpose(),
                            case_uncertain.transpose())
                
                acc.extend([[cols[0]]+[k0]+[mcc,f1,sens,spec,tp,fn,tn,fp,no_call,
                                            clear_ctrl_wins,
                                            clear_ctrl_losses,
                                            ctrl_uncertain,
                                            clear_case_wins,
                                            clear_case_losses,
                                            case_uncertain]
                            for k0,mcc,f1,sens,spec,tp,fn,tn,fp,no_call,
                                    clear_ctrl_wins,
                                    clear_ctrl_losses,
                                    ctrl_uncertain,
                                    clear_case_wins,
                                    clear_case_losses,
                                    case_uncertain in results])
            
            else:
                 results = zip(cols[1:],mcc,f1,sens,spec,tp,fn,tn,fp)
                 acc.extend([[cols[0]]+[k0]+[mcc,f1,sens,spec,tp,fn,tn,fp]
                            for k0,mcc,f1,sens,spec,tp,fn,tn,fp in results])
            # move on to the next column in order to regress all others against it
            case = np.roll(case,-1,axis=1)
            ctrl = np.roll(ctrl,-1,axis=1)
            cols = np.roll(cols,-1)

        self.results = acc

class Discovery ():
    def __init__(self,ct_case,ct_ctrl,ct_other=None):
        self.ct_case = ct_case
        self.ct_ctrl = ct_ctrl
        self.ct_other = ct_other
        self.search = DiscoverySearch(ct_case,ct_ctrl)
        
    def test_all_combos (self, threshold_case=0.5, threshold_control=0.5, find_no_call = False):
        self.search.test_all_combos(threshold_case = threshold_case, threshold_control=threshold_control, find_no_call = find_no_call)            
        self.results = self.search.results
        
    def R2 (self,x,y):
        ''' This calculates the coefficient of determination (R^2) for a regression line, given
        x and y sample values. '''
        params,SSres,_,_,_ = np.polyfit(x,y,1,full=True)
        SSres = SSres[0]
        SStot = np.sum((y - y.mean(axis=0))**2)
        return 1 - (SSres/SStot)  
    
    def initialize_best_pairs (self, sort_by = 'mcc', n_best = 500, find_no_call = False):
        ''' accc
        Sorts self.results by one of 'mcc', 'f1','sens','spec','tp','fn','tn','fp' to
        create self.best_pairs.  The maximum number of pairs to keep in self.best_pairs
        is given by n_best.
        '''
        if not hasattr(self,'results'):
            raise AttributeError("Please execute `run_discovery` before trying to filter results") 

        if find_no_call:
            columns = ['x','y','mcc','f1','sens','spec','tp','fn','tn','fp', 'no_call', 'clear_ctrl_wins', 'clear_ctrl_losses', 'ctrl_uncertain', 'clear_case_wins', 'clear_case_losses', 'case_uncertain']
        else:
            columns = ['x','y','mcc','f1','sens','spec','tp','fn','tn','fp']
        metric2index = {v:k for k,v in enumerate(columns)}
        
        # Initial sort 
        x = sorted(self.results, key=lambda a: a[metric2index[sort_by]], reverse=True)
        # take the n best, create a score column equal to the sorting metric 
        self.best_pairs = pd.DataFrame(x[:n_best],columns=columns)
        self.best_pairs.reset_index(inplace=True,drop=False)
        self.best_pairs['score'] = self.best_pairs[sort_by]
        
    
    def filter_results (self,sort_by = 'mcc',n_best=500,r2_adj = False,loo = False, re_init = False, find_no_call = False):
        ''' 
        Returns the n_best PIRM pairs.  An intial sort is done by `initialize_best_pairs`
        according to one of 'mcc', 'f1','sens','spec','tp','fn','tn','fp'.  This reduces
        the number of best pairs to n_best pairs. 
        
        If r2_adj is True, then self.best_pairs is sorted again after multiplying each pair's 
        score (e.g., mcc, f1, etc.) by the minimum of the R^2 scores for the pair.  This is 
        done to favor PIRMs whose parameters best explain the variance in their data. If loo
        is True, this runs leave-one-out for whichever score we selected (e.g., mcc, f1, etc.)
        
        '''
        # initialize best pairs only if best_pairs doesn't exist or re_init is True
        if not hasattr(self,'best_pairs') or re_init:
            self.initialize_best_pairs (sort_by = sort_by, n_best = n_best, find_no_call=find_no_call) 


        if loo:
            # run leave-one-out for each of the PIRM pairs returning the leave-one-out estimate mcc score
            self.best_pairs['loo_adj'] = [case_ctrl_loo(self.ct_case,self.ct_ctrl,[row.x,row.y],stats = [sort_by])[0] 
                                          for i,row in self.best_pairs.iterrows()]
        if r2_adj:
            # Add the coefficients of determination (R^2) for case and ctrl PIRMs
            self.best_pairs['case_R2'] = self.best_pairs.apply(lambda row: self.R2(self.ct_case.ctp[row.x],self.ct_case.ctp[row.y]),axis=1)
            self.best_pairs['ctrl_R2'] = self.best_pairs.apply(lambda row: self.R2(self.ct_ctrl.ctp[row.x],self.ct_ctrl.ctp[row.y]),axis=1)
            
            # get the minimum R^2
            min_r2 = self.best_pairs[['case_R2','ctrl_R2']].min(axis=1)
            
            if loo:
                # if loo is True, the score is the loo_adj score * min_r2
                self.best_pairs.score = self.best_pairs.loo_adj * min_r2
            else:
                # otherwise it is just the original score adjusted by min_r2
                self.best_pairs.score = self.best_pairs.score * min_r2
            
        # substract 0.2 times no call from score
        if find_no_call:
            self.best_pairs.score = self.best_pairs.score - 0.2 * self.best_pairs.no_call
        self.best_pairs = self.best_pairs.sort_values(by='score',ascending=False).reset_index(drop=True)
        
    def pairs2screens (self, choose=1, out_of=1):
        ''' Makes 2d list of every possible permutation of 'choose' screens out of the top 'out_of' screens'''
        top_screens = [Screen(name=None,ct_case=self.ct_case,ct_ctrl=self.ct_ctrl,pair = self.get_pair(i)) for i in range(out_of)]
        figs = [(self.best_pairs[i:i+1].reset_index(drop=False).iloc[0], self.ct_case, self.ct_ctrl, self.ct_other) for i in range(out_of)]
        return list(combinations(top_screens, choose)), list(combinations(figs, choose))
            
    def get_pair (self, i = 0, full = False):
        ''' If i is an integer, this returns the record for the ith best pair.  If i is a string, this
        returns all pairs in which this string appears as x or y. If full is True this returns the 
        entire record, else just the x and y.'''
        
        if not hasattr(self,'best_pairs'):
            raise AttributeError("Please filter results before ranking pairs")
        0
        if type(i) == int:
            if i > len(self.best_pairs) - 1:
                print(f"\nWARNING: requesting best pair {i} but there are only {len(self.best_pairs)} best pairs.\n")
            
            pair = self.best_pairs.iloc[i]
            return pair if full else [pair.x, pair.y] 
        
        elif type(i) in [str,np.str_]:
            pairs = self.best_pairs[(self.best_pairs.x == i) | (self.best_pairs.y == i)]
            return pairs if full else [[row.x,row.y] for i,row in pairs.iterrows()]
        
        else:
            raise ValueError("get_pair requires its argument to be an integer or a string")
                                    
    
    def rank_pairs (self,n_best=30):
        if not hasattr(self,'best_pairs'):
            raise AttributeError("Please filter results before ranking pairs")
        best_pairs = self.best_pairs[:n_best]
        u,c=np.unique([*best_pairs.x]+[*best_pairs.y],return_counts=True)
        return sorted(dict(zip(u,c)).items(),key=lambda x: x[1],reverse=True)
    
    def posthoc (self):
        '''Deprecated: Use plot_pairs'''
        self.plot_pairs()
        
    def plot_pairs (self, rows = 2, cols = 3, samples=None, figsize=None, find_no_call = False):
        colors = ['magenta','blueviolet','crimson','cyan','purple','orange','black','green','blue','saddlebrown']
        mini = self.best_pairs[:(rows*cols)].reset_index(drop=False)
        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize= figsize or (12, 3*rows))
        
        # Iterate over each subplot
        for i in range(rows):
            for j in range(cols):
                row = mini.iloc[i*cols + j]
                X,Y = row.x,row.y

                cases = np.stack([
                    self.ct_case.ctp[X].to_numpy(),
                    self.ct_case.ctp[Y].to_numpy()
                ]).T
                ctrls = np.stack([
                    self.ct_ctrl.ctp[X].to_numpy(),
                    self.ct_ctrl.ctp[Y].to_numpy()
                ]).T

                x_case,y_case,x_ctrl,y_ctrl = cases[:,0],cases[:,1],ctrls[:,0],ctrls[:,1]

                m_case,b_case = np.polynomial.polynomial.Polynomial.fit(x_case,y_case,1)
                m_ctrl,b_ctrl = np.polynomial.polynomial.Polynomial.fit(x_ctrl,y_ctrl,1)

                if rows > 1:
                    ax=axes[i,j]
                else:
                    if cols > 1:
                        ax = axes[j]
                    else:
                        ax = axes

                if find_no_call:
                    clear_ctrl_wins_x, clear_ctrl_wins_y, clear_ctrl_losses_x, clear_ctrl_losses_y, ctrl_uncertain_x, ctrl_uncertain_y, \
                    clear_case_wins_x, clear_case_wins_y, clear_case_losses_x, clear_case_losses_y, case_uncertain_x, case_uncertain_y = ([] for i in range(12))

                    #populate case value lists
                    # the itr//10 is because there are 10 subsamples, thus row.clear_case_wins has one
                    # true/false value for each of the 10 x_case values
                    for itr, x_case_val in enumerate(x_case):
                        if row.clear_case_wins[itr//10]:
                            clear_case_wins_x.append(x_case_val)
                            clear_case_wins_y.append(y_case[itr])
                        elif row.clear_case_losses[itr//10]:
                            clear_case_losses_x.append(x_case_val)
                            clear_case_losses_y.append(y_case[itr])
                        elif row.case_uncertain[itr//10]:
                            case_uncertain_x.append(x_case_val)
                            case_uncertain_y.append(y_case[itr])
                        if row.clear_case_wins[itr//10] and row.case_uncertain[itr//10]\
                            or row.clear_case_losses[itr//10] and row.case_uncertain[itr//10]\
                            or row.clear_case_losses[itr//10] and row.clear_case_wins[itr//10]:
                            print("ERROR: Overlapping points in case")


                    for itr, x_ctrl_val in enumerate(x_ctrl):
                        if row.clear_ctrl_wins[itr//10]:
                            clear_ctrl_wins_x.append(x_ctrl_val)
                            clear_ctrl_wins_y.append(y_ctrl[itr])
                        elif row.clear_ctrl_losses[itr//10]:
                            clear_ctrl_losses_x.append(x_ctrl_val)
                            clear_ctrl_losses_y.append(y_ctrl[itr])
                        elif row.ctrl_uncertain[itr//10]:
                            ctrl_uncertain_x.append(x_ctrl_val)
                            ctrl_uncertain_y.append(y_ctrl[itr])
                        if row.clear_ctrl_wins[itr//10] and row.ctrl_uncertain[itr//10]\
                            or row.clear_ctrl_losses[itr//10] and row.ctrl_uncertain[itr//10]\
                            or row.clear_ctrl_losses[itr//10] and row.clear_ctrl_wins[itr//10]:
                            print("ERROR: Overlapping points in control")
                        


                    ax.scatter(clear_case_wins_x, clear_case_wins_y,s=3,c='y', label='case win')
                    ax.scatter(clear_ctrl_wins_x, clear_ctrl_wins_y,s=3,c='#1e90ff', label = 'ctrl win')

                    ax.scatter(case_uncertain_x, case_uncertain_y,s=3,c='orange', label= 'case no-call')
                    ax.scatter(ctrl_uncertain_x, ctrl_uncertain_y,s=3,c='#1effff', label = 'ctrl no-call')

                    ax.scatter(clear_case_losses_x, clear_case_losses_y,s=3,c='r', label = 'case loss')
                    ax.scatter(clear_ctrl_losses_x, clear_ctrl_losses_y,s=3,c='#8f1fff', label = 'ctrl loss')
                else:
                    ax.scatter(x_case,y_case, s=3,c='y', label = 'case')
                    ax.scatter(x_ctrl,y_ctrl, s=3,c='dodgerblue', label = 'control')

                #plot slopes
                ax.plot(x_case, x_case * m_case + b_case,c='y',linewidth=.5)
                ax.plot(x_ctrl, x_ctrl * m_ctrl + b_ctrl,c='dodgerblue',linewidth=.5)
                ax.legend( fontsize="7", loc ="lower right")

                print()
                if samples:
                    for _id,color in zip(samples,colors):
                        case_records = self.ct_case.ctp[self.ct_case.ids==_id]
                        ctrl_records = self.ct_ctrl.ctp[self.ct_ctrl.ids==_id]
                        if not case_records.empty:
                            ax.scatter(case_records[X],case_records[Y],s=10,c=color)
                        if not ctrl_records.empty:
                            ax.scatter(ctrl_records[X],ctrl_records[Y],s=10,c=color)

                ax.set_xlabel(X)
                ax.set_ylabel(Y)
                ax.text(.4, .9, f"score: {row.score:.4f}", ha='right', va='bottom', transform=ax.transAxes)
        fig.tight_layout()
        plt.show()