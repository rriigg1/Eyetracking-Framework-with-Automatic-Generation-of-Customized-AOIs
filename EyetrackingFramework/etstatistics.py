import config
import result_classes


def fixations_per_AOI(fixated_aois_data, start_frame=0, end_frame=None):
    """
    Counts the fixations for every occuring aoi for all the FixatedAOIs objects in fixated_aois_data.
    """
    assert isinstance(fixated_aois_data, result_classes.FixatedAOIsData)
    fixated_aois = fixated_aois_data.aoi_data
    aoi_fix_counts = result_classes.FixationCounts()
    for data in fixated_aois:
        if (end_frame is None):
            cur_end = len(data.aois)
        else:
            cur_end = end_frame
        aois = list(filter(lambda a: start_frame <= a[0] <= cur_end, data.aois))
        aoi_fix_count = {}
        for aoi_name in data.aoi_names:
            aoi_fix_count[aoi_name] = 0
        aoi_fix_count["NONE"] = 0
        for aoi in aois:
            if (aoi[1] in aoi_fix_count):
                aoi_fix_count[aoi[1]] += 1
            else:
                aoi_fix_count[aoi[1]] = 1
        aoi_fix_counts.append_count(aoi_fix_count, data.video_file,
                                    participant=data.participant,
                                    source=data.source,
                                    info=data.info,
                                    grouping=data.grouping)
    return aoi_fix_counts


def saccades_between_AOIs(fixated_aois_data, filter_equal=True, start_frame=0, end_frame=None):
    """
    Counts the saccades between all the aois.
    Args:
        fixated_aois_data: FixatedAOIsData
            Contains the fixated aois for a given video and participant.
        filter_equal: bool
            If True only counts saccades between two different aois.
    """
    assert isinstance(fixated_aois_data, result_classes.FixatedAOIsData)
    fixated_aois = fixated_aois_data.aoi_data
    aoi_saccade_counts = result_classes.AOISaccadesData()
    for data in fixated_aois:
        if (end_frame is None):
            cur_end = len(data.aois)
        else:
            cur_end = end_frame
        aois = list(filter(lambda a: start_frame <= a[0] <= cur_end, data.aois))
        aoi_saccade_count = {}
        last_aoi = aois[0][1]
        for aoi in aois[1:]:
            if ((last_aoi, aoi[1]) in aoi_saccade_count):
                aoi_saccade_count[(last_aoi, aoi[1])] += 1
            else:
                aoi_saccade_count[(last_aoi, aoi[1])] = 1
            last_aoi = aoi[1]
        if (filter_equal):
            aoi_saccade_count = {key: aoi_saccade_count[key] for key in aoi_saccade_count if key[0] != key[1]}
        aoi_saccade_counts.append_count(aoi_saccade_count, data.video_file,
                                        participant=data.participant,
                                        source=data.source,
                                        info=data.info,
                                        grouping=data.grouping)
    return aoi_saccade_counts
