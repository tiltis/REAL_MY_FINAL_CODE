function out_rgb = runProposed(hdr,Lmax)
cd ./proposed_ag
out_rgb = iCAM06_HDR_Proposed(hdr,Lmax);
cd ../