# -*-coding:utf-8-*-
import numpy as np
import torch


def ScanMatch_batch(seq_preds, gts, valid_lens, ScanMatchInfo_):
    sample_num = seq_preds.size(1)
    seq_preds[:, :, :, 0] *= 12
    seq_preds[:, :, :, 1] *= 12
    seq_preds = torch.clamp(seq_preds, max=11)
    seq_preds_num = seq_preds[:, :, :, 0] + seq_preds[:, :, :, 1] * 12
    gts[:, :, 0] *= 12
    gts[:, :, 1] *= 12
    gts = torch.clamp(gts, max=11)
    gts_num = gts[:, :, 0] + gts[:, :, 1] * 12
    score_all = list()
    for num in range(sample_num):
        scores = ScanMatch_compute_batch(seq_preds_num[:, num], gts_num, valid_lens, ScanMatchInfo_)
        score_all.append(scores.unsqueeze(-1))
    return torch.cat(score_all, dim=-1)


def ScanMatch_compute_batch(seq_preds, gts, valid_lens, ScanMatchInfo_):
    batch = seq_preds.size(0)
    n = gts.size(1)
    m = seq_preds.size(1)
    ScoringMatrix = ScanMatchInfo_['SubMatrix'][0][0]
    gap = ScanMatchInfo_['GapValue']
    # ScoringMatrix = torch.from_numpy(ScoringMatrix).to(seq_preds.device)
    gap = 0
    best_matrix = torch.zeros(batch, n + 1, m + 1)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                best_matrix[:, i, j] = gap * j
            elif j == 0:
                best_matrix[:, i, j] = gap * i
            else:
                match = ScoringMatrix[gts[:, i-1].int().cpu().numpy(), seq_preds[:, j-1].cpu().int().numpy()]

                gap1_score = best_matrix[:, i - 1, j] + gap
                gap2_score = best_matrix[:, i, j - 1] + gap
                match_score = best_matrix[:, i - 1, j - 1] + match
                score = torch.stack((gap1_score, gap2_score, match_score), dim=0)
                best_matrix[:, i, j] = score.max(dim=0)[0]
    scores = torch.zeros(batch)

    for index in range(batch):
        score = best_matrix[index, :valid_lens[index]].max()
        max_sub = ScoringMatrix.max()
        scale = max_sub * max(m, n)
        score = score / scale
        scores[index] = score

    return scores
