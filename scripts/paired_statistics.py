"""Offline paired tests; undefined/insufficient statistics are explicit nulls."""
import math
import statistics


def analyze(values):
    from scipy import stats
    values=[float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite paired difference')
    n=len(values)
    out=dict(n=n,mean=statistics.mean(values) if values else None,std=statistics.stdev(values) if n>1 else None,
             ci95=None,paired_t=None,wilcoxon=None,sign_test=None)
    if n>1:
        sd=out['std'];half=float(stats.t.ppf(.975,n-1))*sd/math.sqrt(n)
        out['ci95']=[out['mean']-half,out['mean']+half]
        if sd:
            result=stats.ttest_1samp(values,0.)
            out['paired_t']=dict(statistic=float(result.statistic),pvalue=float(result.pvalue),alternative='two-sided')
        else:out['paired_t']=dict(statistic=None,pvalue=1. if out['mean']==0 else 0.,degenerate_constant_difference=True)
        if any(v!=0 for v in values):
            result=stats.wilcoxon(values,alternative='two-sided',zero_method='wilcox',method='auto')
            out['wilcoxon']=dict(statistic=float(result.statistic),pvalue=float(result.pvalue),alternative='two-sided')
        else:out['wilcoxon']=dict(statistic=0.,pvalue=1.,all_zero=True)
    nonzero=[v for v in values if v!=0]
    if nonzero:
        result=stats.binomtest(sum(v>0 for v in nonzero),len(nonzero),.5,alternative='two-sided')
        out['sign_test']=dict(positive=sum(v>0 for v in nonzero),n=len(nonzero),pvalue=float(result.pvalue),zeros_excluded=n-len(nonzero))
    return out
