"""Conservative complete-link grouping for human-reviewed duplicate candidates."""

def complete_link_groups(names, scores, threshold):
    """Only merge clusters when every proposed cross-pair meets the threshold."""
    clusters=[{i} for i in range(len(names))]
    def score(i,j):
        if isinstance(scores,dict): return float(scores[(min(i,j),max(i,j))])
        return float(scores[min(i,j)][max(i,j)])
    while True:
        candidates=[]
        for a in range(len(clusters)):
            for b in range(a+1,len(clusters)):
                cross=[score(i,j) for i in clusters[a] for j in clusters[b]]
                if cross and min(cross)>=threshold:
                    candidates.append((min(cross),a,b))
        if not candidates: break
        _,a,b=max(candidates,key=lambda item:(item[0],-item[1],-item[2]))
        clusters[a]|=clusters[b]; del clusters[b]
    groups=[]; assigned=set()
    for members in sorted((c for c in clusters if len(c)>1),key=lambda c:min(c)):
        ordered=sorted(members); assigned.update(ordered)
        groups.append({"files":[names[i] for i in ordered],
                       "pairs":[{"a":names[i],"b":names[j],"similarity":score(i,j)}
                                for pos,i in enumerate(ordered) for j in ordered[pos+1:]]})
    pairwise=[{"a":names[i],"b":names[j],"similarity":score(i,j),"above_threshold":score(i,j)>=threshold}
              for i in range(len(names)) for j in range(i+1,len(names))]
    grouped_pairs={frozenset((p["a"],p["b"])) for group in groups for p in group["pairs"]}
    review_pairs=[p for p in pairwise if p["above_threshold"] and frozenset((p["a"],p["b"])) not in grouped_pairs]
    return {"groups":groups,"pairwise_scores":pairwise,"manual_review_pairs":review_pairs,
            "unmatched":[names[i] for i in range(len(names)) if i not in assigned],
            "grouping_policy":"complete_link_all_pairwise_scores_must_meet_threshold"}
