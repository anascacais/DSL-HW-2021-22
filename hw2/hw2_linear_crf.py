import torch
from torch import nn


class LinearChainCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        num_classes (int): number of classes in your tagset.
        pad_index (int, optional): integer representing the pad symbol in your tagset.
            If not None, the model will apply constraints for PAD transitions.
            NOTE: there is no need to use padding if you use batch_size=1.
    """

    def __init__(self, num_classes, pad_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes))
        self.initials = nn.Parameter(torch.randn(self.num_classes))
        self.finals = nn.Parameter(torch.randn(self.num_classes))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.initials, -0.1, 0.1)
        nn.init.uniform_(self.finals, -0.1, 0.1)

        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero
        if self.pad_index is not None:
            # no transitions from padding
            self.transitions.data[self.pad_index, :] = -10000.0
            # no transitions to padding
            self.transitions.data[:, self.pad_index] = -10000.0
            # except if we are in a pad position
            self.transitions.data[self.pad_index, self.pad_index] = 0.0

    def forward(self, emissions, mask=None):
        """Run the CRF layer to get predictions."""
        return self.decode(emissions, mask=mask)

    def neg_log_likelihood(self, emissions, tags, mask=None):
        """
        Compute the negative log-likelihood of a sequence of tags given a sequence of
        emissions scores.
        
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels)
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len)
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len)
        
        Returns:
             torch.Tensor: the sum of neg log-likelihoods of each sequence in the batch.
                Shape of ([])
        """
        scores = self.compute_scores(emissions, tags, mask=mask)
        partition = self.compute_log_partition(emissions, mask=mask)
        nll = -torch.sum(scores - partition) 
        return nll

    def decode(self, emissions, mask=None):
        """
        Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels).
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len)
        
        Returns:
            list of lists: the best viterbi sequence of labels for each batch.
        """
        scores, viterbi_path = self.viterbi(emissions, mask=mask)
        return viterbi_path, scores

    def compute_scores(self, emissions, tags, mask=None):
        """
        Compute the scores for a given batch of emissions with their tags.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """

        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)

        # transition score from START to the first tags for each batch
        t_scores = self.transitions[0, tags[:, 0]]

        # emisson score for first character
        e_scores = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze()

        # score for first character
        scores += e_scores + t_scores

        # loop over remaining characters (except last one)
        for i in range(1, seq_length):

            previous_tags = tags[:, i-1]
            current_tags = tags[:, i]

            # emission and transition scores
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            scores += e_scores + t_scores

        # transition score from last tags to STOP for each batch
        scores += self.transitions[tags[:, -1], -1]
        #print(f'scores: {scores}')

        return scores


    def compute_log_partition(self, emissions, mask=None):
        """
        Compute the partition function in log-space using the forward-algorithm.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """

        batch_size, seq_length, nb_labels = emissions.shape
        forward = self.transitions[0, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):

            tag_forward = []

            for tag in range(nb_labels):

                # emission score for current tag
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # transitions scores for current tag
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                scores = e_scores + t_scores + forward
                tag_forward.append(torch.logsumexp(scores, dim=1))

            # create tensor
            forward = torch.stack(tag_forward).t()

        # add the scores for the final transition
        last_transition = self.transitions[:, -1]
        end_scores = forward + last_transition.unsqueeze(0)

        #print(f'log partition: {torch.logsumexp(end_scores, dim=1)}')

        return torch.logsumexp(end_scores, dim=1)

    def viterbi(self, emissions, mask=None):
        """
        Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        
        Returns:
            list of lists of ints: the best viterbi sequence of labels for each batch
        """

        batch_size, seq_length, nb_labels = emissions.shape
        sequences = []

        # score for first character
        forward = self.transitions[0, :].unsqueeze(0) + emissions[:, 0]
        
        # Forward pass.
        for i in range(1, seq_length):

            tag_forward = []
            tag_sequence = []

            for tag in range(nb_labels):
                
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + forward

                # get tag with highest score
                max_score, max_score_tag = torch.max(scores, dim=-1)
                tag_forward.append(max_score)
                tag_sequence.append(max_score_tag)

            forward = torch.stack(tag_forward).t()
            sequences.append(tag_sequence)

        # get tag with highest score
        end_scores = forward + self.transitions[:, -1].unsqueeze(0)
        paths_scores, max_final_tags = torch.max(end_scores, dim=1)
        #print(f'max final tags: {max_final_tags}')

        # backward pass.
        best_paths = []

        for i in range(batch_size):

            # most likely last character
            final_tag = max_final_tags[i].item()

            best_tag = final_tag
            best_path = [final_tag]

            for j in range(seq_length - 2, -1, -1):

                # most likely character for position j
                best_tag = sequences[j][best_tag][i].item()
                best_path.insert(0, best_tag)

            best_paths.append(best_path)

        #print(f'best paths: {best_paths}')

        return paths_scores, best_paths
