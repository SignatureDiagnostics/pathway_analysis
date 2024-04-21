from PIRMDiscovery import *

class BestPIRMPairs:
    """
    Uses PIRMDiscovery to create sets of IDs, as well as run the regression model using PIRMDiscovery
    """
    def __init__(self, pattern_probs, sd):
        rng = np.random.default_rng(seed=82454) # stuff to make random sampling replicable
        self.pattern_probs = pattern_probs

        # Not all births at less than 37 weeks are classified as preterm, so add a column for that
        sd['EarlyBirth'] = sd['weeks gestation at delivery (column AN)'] <= 36

        # The samples we want:

        tri21_ids = list(sd[sd['Fetal Condition'] == 'Trisomy 21'].Sample_ID)
        tri18_ids = list(sd[sd['Fetal Condition'] == 'Trisomy 18'].Sample_ID)
        tri13_ids = list(sd[sd['Fetal Condition'] == 'Trisomy 13'].Sample_ID)
        monox_ids = list(sd[sd['Fetal Condition'] == 'Monosomy X'].Sample_ID)

        control_ids = list(
            sd[(sd['Fetal Condition'].isna()) & 
            (sd['Maternal Condition'].isna())
            ].Sample_ID)

        # check that we have data for these samples
        self.tri13_ids   = [_id for _id in tri13_ids if self.check_for_data(_id,'chr1','chr13')]
        self.tri18_ids   = [_id for _id in tri18_ids if self.check_for_data(_id,'chr1','chr18')]
        self.tri21_ids   = [_id for _id in tri21_ids if self.check_for_data(_id,'chr1','chr21')]
        self.monox_ids   = [_id for _id in monox_ids if self.check_for_data(_id,'chrX','chr1')]
        self.control_ids = [_id for _id in control_ids if self.check_for_data(_id,'chr1','chr21','chr18','chr13')]



        # The samples we want for preterm/preeclampsia:
        preterm_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                      (sd['Maternal Condition'] == 'Preterm Birth')].Sample_ID)

        pe_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                            ((sd['Maternal Condition'] == 'Mild Preeclampsia') |
                            (sd['Maternal Condition'] == 'Severe Preeclampsia') )
                            ].Sample_ID)

        pes_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                        (sd['Maternal Condition'] == 'Severe Preeclampsia') 
                            ].Sample_ID)

        pem_ids = list(sd[(sd['Fetal Condition'].isna()) &
                        (sd['Maternal Condition'] == 'Mild Preeclampsia')
                            ].Sample_ID)

        # Early birth is the controls plus the mild preeclampsia
        earlyterm_ids = list(sd[sd['EarlyBirth'] == True].Sample_ID)
        usualterm_ids = list(sd[sd['EarlyBirth'] == False].Sample_ID)


        # check that we have data for these samples
        self.preterm_ids   = [_id for _id in preterm_ids if self.check_for_data(_id,'chr1')]
        self.pe_ids   = [_id for _id in pe_ids if self.check_for_data(_id,'chr1')]
        self.pem_ids   = [_id for _id in pem_ids if self.check_for_data(_id,'chr1')]
        self.pes_ids   = [_id for _id in pes_ids if self.check_for_data(_id,'chr1')]


        # Male and female control samples (no maternal or fetal conditions):
        female_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                            (sd['Maternal Condition'].isna()) &
                            (sd['infant sex (column AL)']=='F')].Sample_ID)

        male_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                        (sd['Maternal Condition'].isna()) &
                        (sd['infant sex (column AL)']=='M')].Sample_ID)

        self.female_control_ids   = [_id for _id in female_ids if self.check_for_data(_id)]
        self.male_control_ids   = [_id for _id in male_ids if self.check_for_data(_id,'chr1')]


        # Male and female samples (no fetal conditions but includes PE and PTB):
        female_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                            (sd['infant sex (column AL)']=='F')].Sample_ID)

        male_ids = list(sd[(sd['Fetal Condition'].isna()) & 
                        (sd['infant sex (column AL)']=='M')].Sample_ID)

        self.female_ids   = [_id for _id in female_ids if self.check_for_data(_id)]
        self.male_ids   = [_id for _id in male_ids if self.check_for_data(_id,'chr1')]

        # Early birth is the controls plus the mild preeclampsia
        self.earlyterm_ids =  [_id for _id in earlyterm_ids if self.check_for_data(_id)]
        self.usualterm_ids =  [_id for _id in usualterm_ids if self.check_for_data(_id)]

        self.condition_ids = [self.male_ids,self.female_ids,self.control_ids, self.tri13_ids, self.tri18_ids, self.tri21_ids, self.monox_ids, self.pe_ids, self.preterm_ids, self.pe_ids+self.preterm_ids]
        self.condition = ['male','female','control','tri13','tri18','tri21','monox','pe','preterm','pe_ptb']

        self.group_size = {c:len(c_id) for c,c_id in zip(self.condition,self.condition_ids)}

        self.chroms = ['chr1','chr2','chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
                'chr20', 'chr21','chrX','chrY']

        self.patterns = ['1','0','11','00','111','000']  # mixed patterns rarely make it into the best models
        
    def check_for_data (self, sample_id, *regions):
        ''' Checks that we have pattern probability information for the given
        sample id and region.'''
        pp = self.pattern_probs.get(sample_id)
        return False if pp is None else all([pp.get(region) is not None for region in regions])

    def run_one_test (self, case_label, case_ids, ctrl_label = None, ctrl_ids = None, other_ids=None, n_subsamples = 10, 
         sort_by = 'mcc', report=True, plot=True, threshold_case=0.5, threshold_control = 0.5, find_no_call = False):
        ''' Runs a case-ctrl test, prints and plots results, returns the Discovery object. '''
        
        # Only use X and Y chromosomes for testing sex and monosomy x
        _chroms = ['chr1','chr2','chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 
                'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21']
        
        if case_label in ['monox','male','female'] or ctrl_label in ['monox','male','female']:
            _chroms = _chroms + ['chrX','chrY']
            
        
        # a contrast in which the second element is not our control_ids samples
        if ctrl_label is not None and ctrl_ids is not None:
            _ct_ctrl = CT(
                pattern_probs = self.pattern_probs,
                sample_ids = ctrl_ids, 
                group_label = ctrl_label, 
                regions = _chroms, 
                patterns = self.patterns,
                n_subsamples = n_subsamples,
                postprocess = hypo_hyper_mixed, 
                N = 30000)
        else:
            _ct_ctrl = CT(
                pattern_probs = self.pattern_probs,
                sample_ids = self.control_ids, 
                group_label = 'control', 
                regions = _chroms, 
                patterns = self.patterns,
                n_subsamples = n_subsamples,
                postprocess = hypo_hyper_mixed, 
                N = 30000)

        if report:
            print(f"\nTesting {case_label} samples against {ctrl_label or 'control'} samples\n================= n = {len(case_ids)} =====================")
        
        # the case samples
        ct_case = CT(
            pattern_probs = self.pattern_probs,
            sample_ids = case_ids, 
            group_label = case_label, 
            regions = _chroms, 
            patterns = self.patterns,
            n_subsamples = n_subsamples,
            postprocess = hypo_hyper_mixed, 
            N = 30000)
        
        if other_ids is not None:
            ct_other = CT(
                pattern_probs = self.pattern_probs,
                sample_ids = other_ids, 
                group_label = 'multi_group', 
                regions = _chroms, 
                patterns = self.patterns,
                n_subsamples = n_subsamples,
                postprocess = hypo_hyper_mixed, 
                N = 30000)
        else:
            ct_other = None
        
        
        d = Discovery(ct_case,_ct_ctrl,ct_other)

        #this calculates which ones are certainly case, certainly control, and uncertain
        d.test_all_combos(threshold_case = threshold_case, threshold_control = threshold_control, find_no_call = find_no_call)
        if len(case_ids) > 2:
            d.filter_results(n_best=50,loo=True,r2_adj=True,sort_by=sort_by, find_no_call = find_no_call)
            if report:
                if find_no_call:
                    print(f"Best pairs\n{d.best_pairs[:6][['x','y','mcc','sens','spec','tp','fn','fp','tn','case_R2','ctrl_R2','loo_adj', 'no_call', 'score']]}")
                else:
                    print(f"Best pairs\n{d.best_pairs[:6][['x','y','mcc','sens','spec','tp','fn','fp','tn','case_R2','ctrl_R2','loo_adj', 'score']]}")

        else:
            d.filter_results(n_best=50,sort_by=sort_by, find_no_call = find_no_call)
            if report:
                if find_no_call:
                    print(f"Best pairs\n{d.best_pairs[:6][['x','y','mcc','sens','spec','tp','fn','fp','tn','case_R2','ctrl_R2','no_call', 'score']]}")
                else:
                    print(f"Best pairs\n{d.best_pairs[:6][['x','y','mcc','sens','spec','tp','fn','fp','tn','case_R2','ctrl_R2', 'score']]}")
        
        if plot:
            d.plot_pairs(rows=3,cols=3, find_no_call = find_no_call)
        return d