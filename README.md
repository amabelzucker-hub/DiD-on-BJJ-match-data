# DiD-on-BJJ-match-data
This project uses both staggered and standard Difference in Differences approaches in CausalPy to perform causal inference analyses on BJJ Match data. 

Most of the model implementation is commented out to so I can de-bug the dataframe handling and then test models with it one at a time. I jsut sub out the appropriate variables for the implementations to test different analyses. 

Bug: staggered DiD had good paralell trends but highly variable and large confidence interavls post-treatment earlier in project development, before I tried bins, filtered for pre and post treament observations on the data, and meddled with it in other ways to make it usable for the standard DiD model. So, I would like to go through and possibly separate out at some point the dataframe I use for each so I don't have negative interactions like this. 

Bug: I should have 43 treated fighters in the 2022 treatment group dataframe AFTER filtering for, each with post_treatment == 1 for post-treatment observations, but the sum of post-treatment where treated == 1 in the frame is only 15. Clear mishandling of data somewhere in there. 

Issue: unsure if for standard DiD model post_treatment should be 0 for ALL control observations, or if it should be 1 on observations past the treatment time of 2022. 

Half-worked parts: filtering daatframe to use only submission data but still maintaining that each fighter has pre and post treatment observations, then using that daataframe to understand causal impacts on the frequency of specific submissions post treatment relative to other submissions
