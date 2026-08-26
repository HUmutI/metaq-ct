"""TotalSegmentator ROI names for the pediatric 13-region schema.

Region ids 1-10 are frozen identical to the adult schema so anatomy query rows
transfer verbatim and existing CT-RATE masks stay valid. 11/12/13 are the
pediatric additions.

Measured cost of the extra structures: 59.2 s/volume vs 35.7 s for the adult
subset (1.66x) - region 13 is free because it lives in the same two model parts
as the adult set, region 11 is free because it is derived from the lungs in
prepare_data, and only region 12 pulls in the vertebrae/muscles/ribs parts.
"""

# regions 1-10, unchanged from arc-ct's FINE_LABEL_MAP
ADULT = [
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",      # 1-5
    "trachea",                                              # 6
    "heart", "atrial_appendage_left",                       # 7
    "aorta",                                                # 8
    "pulmonary_vein", "brachiocephalic_trunk",
    "superior_vena_cava", "inferior_vena_cava",             # 9
    "esophagus",                                            # 10
]

# region 13 - upper abdomen. Same model parts as ADULT, so it costs nothing.
# Small bowel/duodenum/colon deliberately excluded: mostly outside a chest FOV
# and they would drag the region centroid caudally out of the 96-slice crop.
UPPER_ABDOMEN = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
    "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left",
    "portal_vein_and_splenic_vein",
]

# region 12 - chest wall and thoracic skeleton. This is the only group that
# costs extra: it pulls in the vertebrae, muscles and ribs model parts.
# Cervical vertebrae and humerus are included because pediatric scans often
# extend above the thoracic inlet; drop them if the FOV analysis says otherwise.
CHEST_WALL = (
    ["sternum", "costal_cartilages",
     "autochthon_left", "autochthon_right",
     "scapula_left", "scapula_right",
     "clavicula_left", "clavicula_right",
     "humerus_left", "humerus_right"]
    + ["rib_left_%d" % i for i in range(1, 13)]
    + ["rib_right_%d" % i for i in range(1, 13)]
    + ["vertebrae_T%d" % i for i in range(1, 13)]
    + ["vertebrae_C%d" % i for i in range(1, 8)]
)

PEDS = ADULT + UPPER_ABDOMEN + CHEST_WALL

# TS v2 id -> pediatric region id. Region 11 has no entry: it is derived from
# the union of 1-5 by dilation in prepare_data, because TotalSegmentator has no
# pleura class and routing effusion/pneumothorax to the lung mask would exclude
# exactly the voxels the query needs.
FINE_LABEL_MAP_PEDS = {
    10: 1, 11: 2, 12: 3, 13: 4, 14: 5,
    16: 6,
    51: 7, 61: 7,
    52: 8,
    53: 9, 54: 9, 62: 9, 63: 9,
    15: 10,
    # 12 chest wall + skeleton
    **{i: 12 for i in range(92, 116)},      # ribs
    116: 12, 117: 12,                       # sternum, costal cartilages
    **{i: 12 for i in range(27, 51)},       # vertebrae
    69: 12, 70: 12, 71: 12, 72: 12, 73: 12, 74: 12,
    86: 12, 87: 12,                         # autochthon
    # 13 upper abdomen
    1: 13, 2: 13, 3: 13, 4: 13, 5: 13, 6: 13, 7: 13, 8: 13, 9: 13, 64: 13,
}


# --------------------------------------------------------------------------- #
# Pediatric 10-region schema (measured, not assumed)
# --------------------------------------------------------------------------- #
# Survival at the 12x12x12 feature grid over 8816 volumes showed 6 of the 10
# adult regions below 98%: vessels 82.6%, esophagus 83.3%, trachea 89.3%,
# right middle lobe 89.9%, aorta 91.0%, heart 93.6%. An empty region does not
# raise - anatomy_qformer turns it into an unrestricted global query - so those
# queries silently stop being anatomy queries on 1 in 5 pediatric scans.
#
# Merging was then measured rather than assumed:
#   trachea+esophagus      97.2%   (still short - they fail on the SAME small
#                                   children, so merging correlated failures
#                                   buys little)
#   heart+aorta+vessels    99.1%   (they fail on DIFFERENT volumes - this works)
#   all five merged        99.8%
#
# The five-way merge is rejected despite being highest: Peribronchial thickening
# and Bronchiectasis route to the airway, and putting the trachea in the same
# region as the heart would let a bronchial-wall query attend to myocardium.
# 0.6 points is not worth an anatomically wrong prior.
FINE_LABEL_MAP_PEDS10 = {
    10: 1, 11: 2, 12: 3, 13: 4, 14: 5,          # 1-5 lung lobes (unchanged ids)
    16: 6, 15: 6,                                # 6 central airway + esophagus
    51: 7, 61: 7, 52: 7, 53: 7, 54: 7, 62: 7, 63: 7,   # 7 heart + great vessels
    # 8 pleural_space: derived from the lungs, no TS class
    **{i: 9 for i in range(92, 116)},            # 9 chest wall + skeleton
    116: 9, 117: 9,
    **{i: 9 for i in range(27, 51)},
    69: 9, 70: 9, 71: 9, 72: 9, 73: 9, 74: 9, 86: 9, 87: 9,
    1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 64: 10,  # 10 upper abdomen
}

PEDS10_NAMES = {
    1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left",
    3: "lung_upper_lobe_right", 4: "lung_middle_lobe_right",
    5: "lung_lower_lobe_right",
    6: "central_airway_and_esophagus",
    7: "heart_and_great_vessels",
    8: "pleural_space",
    9: "chest_wall_and_skeleton",
    10: "upper_abdomen",
}
PEDS10_PLEURA_ID = 8
