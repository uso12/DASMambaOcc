# DASMambaOcc Gap-Focused Design

Key gaps addressed:
- DAOcc: weak explicit temporal smoothing and no structured refinement sub-head
- STCOcc: temporal stability strong but sparse miss risk on small objects
- ALOcc: strong lifting/geometry improvements but heavy full-stack complexity
- OccMamba: strong global modeling but full replacement is high integration risk

Chosen strategy:
- keep DAOcc base pipeline stable
- add adaptive-lifting in vtransform (camera-conditioned + adaptive view weighting + geometry denoise)
- add lightweight temporal memory and detector-guided suppression
- add OccMamba-inspired refinement as a sub-head (not replacing base occupancy head)
