import torch
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import average_precision_score

@torch.no_grad()
def test(model,
        data,
        # head_index: Tensor ,
        # rel_type: Tensor,
        # tail_index: Tensor,
        batch_size: int = 20000,
        k: int = 10,
        log: bool = True,
        getMap: bool = False,
    ) -> Tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        model.eval()
        head_index=data.edge_index[0]
        rel_type=data.edge_type
        tail_index=data.edge_index[1]

        arange = range(torch.numel(head_index))
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k, average_precisions_sk, average_precisions_scr = [], [], [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            # evaluate (s, r, ?)
            if 0:
                scores = []
                tail_indices = torch.arange(model.num_nodes, device=t.device)
                # print("\ntail_indices.split: ", len(tail_indices.split(batch_size)))
                for ts in tail_indices.split(batch_size):
                    # if i == 1:
                        # print(f"\nts: {ts}, type: {type(ts)}, len: {len(ts)}, shape: {ts.shape}")
                        # print(f'h: {h}')
                        # print(f'\nh.expand_as(ts): {h.expand_as(ts)}')

                    scores.append(model(h.expand_as(ts), r.expand_as(ts), ts))      # get scores from model: (s, r, ?)
                    # scores.append(model(ts, r.expand_as(ts), t.expand_as(ts)))      # get scores from model: (?, r, o)
                rank = int((torch.cat(scores).argsort(
                    descending=True) == t).nonzero().view(-1))
                mean_ranks.append(rank)
                reciprocal_ranks.append(1 / (rank + 1))
                hits_at_k.append(rank < k)

            # evaluate (?, r, o)
            scores = []
            tail_indices = torch.arange(model.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(model(ts, r.expand_as(ts), t.expand_as(ts)))      # get scores from model: (?, r, o)
            rank = int((torch.cat(scores).argsort(
                descending=True) == h).nonzero().view(-1))
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

            if getMap:
                # # Calculate precision at each rank and average for MAP from scratch
                # precisions = [int(idx == t) for idx in torch.cat(scores).argsort(descending=True)]
                # precisions_at_rank = [sum(precisions[:i+1]) / (i+1) for i in range(rank+1)]
                # mean_average_precision_scr.append(sum(precisions_at_rank) / len(precisions_at_rank))

                # Calculate MAP using sklearn
                true_labels = (tail_indices == t).float()  # Binary labels
                # print(f"\ntrue_labels: {true_labels.cpu().numpy()}, shape:{true_labels.cpu().numpy().shape} sum: {torch.sum(true_labels.cpu())}")
                # print(f"scores: {scores[0]}, len_shape: {len(scores[0])} shape: {scores[0].shape}")
                map_score = average_precision_score(true_labels.cpu().numpy(), scores[0].cpu().numpy())
                average_precisions_sk.append(map_score)

        print(f'\nlen_hits: {len(hits_at_k)}, len_rr: {len(reciprocal_ranks)}')
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        # print(f"\nmean_ranks = {mean_ranks}\nlen mean_ranks: {len(mean_ranks)}\n max_rank = {max(mean_ranks)} model.num_nodes={model.num_nodes} ")

        map_sk, map_scr = -1, -1
        if getMap:
            map_sk = float(torch.tensor(average_precisions_sk, dtype=torch.float).mean())

            # Calculate precision at each rank and average for MAP
            # precisions = [int(rank < i) / i for i in range(1, k+1)]
            # mean_average_precision_scr.append(sum(precisions) / len(precisions))
            # map_scr = float(torch.tensor(mean_average_precision_scr, dtype=torch.float).mean())

            map_scr = float(torch.tensor(average_precisions_scr, dtype=torch.float).mean())


        return mean_rank, mrr, hits_at_k, map_sk, map_scr

