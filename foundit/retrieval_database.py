import os
import torch
import numpy as np

import faiss
import asmk.index

class FaissCpuL2Index(asmk.index.FaissL2Index):
    def __init__(self, gpu_id):
        super().__init__()
        self.gpu_id = gpu_id

    def _faiss_index_flat(self, dim):
        """Return initialized faiss.IndexFlatL2"""
        return faiss.IndexFlatL2(dim)

asmk.index.FaissGpuL2Index = FaissCpuL2Index

from asmk import io_helpers, asmk_method


class RetrievalDatabase():
    def __init__(self, device="cuda"):

        self.asmk = None
        self.load_asmk()

        self.ivf_builder = self.asmk.create_ivf_builder()

        self.kf_counter = 0
        self.kf_ids = []

        self.query_dtype = torch.float32
        self.query_device = device
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        )

    def load_asmk(self):
        RETRIEVAL_URL = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        CODEBOOK_URL = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"

        hub_dir = torch.hub.get_dir()
        cache_dir = os.path.join(hub_dir, 'checkpoints')
        os.makedirs(cache_dir, exist_ok=True)

        ckpt_path = os.path.join(cache_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
        codebook_path = os.path.join(cache_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")

        if not os.path.isfile(ckpt_path):
            torch.hub.download_url_to_file(RETRIEVAL_URL, ckpt_path)
        if not os.path.isfile(codebook_path):
            torch.hub.download_url_to_file(CODEBOOK_URL, codebook_path)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ckpt_args = ckpt['args']

        asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                        'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                        'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                        'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                        'search': {'topk': None},
                                        'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}
        asmk_params['train_codebook']['codebook']['size'] = ckpt_args.nclusters
        self.asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)
        self.asmk = self.asmk.train_codebook(None, cache_path=codebook_path)


    def prep_features(self, backbone_feat):
        # retrieval_model = self.model

        # Shape of backbone_feat: torch.Size([1, num_tokens, token_size])

        # proj_feat = retrieval_model.projector(backbone_feat) # Dominic: single layer projector, TODO see if we need the projector

        attention = backbone_feat.norm(dim=-1)

        nfeat = 300
        _, topk_indices = torch.topk(attention, min(nfeat, attention.size(1)), dim=1)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, backbone_feat.size(2))
        topk_features = torch.gather(backbone_feat, 1, topk_indices_expanded)

        return topk_features

    def update(self, feat, add_after_query):
        print("Original feature shape: ", feat.shape)
        feat = self.prep_features(feat)
        print("Prepared features: ", feat.shape)
        id = self.kf_counter  # Using own counter since otherwise messes up IVF

        feat_np = feat[0].cpu().numpy()  # Assumes one frame at a time!
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        database_size = self.ivf_builder.ivf.n_images
        # print("Database size: ", database_size, self.kf_counter)

        # Only query if already an image
        topk_image_inds = []
        topk_codes = None  # Change this if actualy querying
        if self.kf_counter > 0:
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)

            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            topk_images = torch.topk(scores, database_size)

            topk_image_inds = topk_images.indices
            topk_image_inds = topk_image_inds.tolist()

        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds, scores[topk_image_inds] if len(topk_image_inds) > 0 else None

    # The reason we need this function is becasue kernel and inverted file not defined when manually updating ivf_builder
    def query(self, feat, id):
        print("Querying database...")
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    def add_to_database(self, feat_np, id_np, topk_codes):
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(self, qvecs, params):
        # Using trick for efficient distance matrix
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        """Accumulate scores for every query image (qvecs, qimids) given codebook, kernel,
        inverted_file and parameters."""
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        for imid, seq in slices:
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            quantized = (qvecs, topk_inds)

            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        """Add descriptors and cooresponding image ids to the IVF

        :param np.ndarray vecs: 2D array of local descriptors
        :param np.ndarray imids: 1D array of image ids
        :param bool progress: step at which update progress printing (None to disable)
        """
        ivf_builder = self.ivf_builder

        step_params = self.asmk.params.get("build_ivf")

        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            # Reuse previously calculated! Only take top 1
            # NOTE: Assuming build params multiple assignment is less than query
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        quantized = (vecs, topk_inds, imids)

        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)


if __name__ == "__main__":

    retrieval_database = RetrievalDatabase()

    token2 = np.load('/home/dominic/vggt/tokens1.npy')
    # print("Token1 shape: ", token1.shape)
    token1 = np.load('/home/dominic/vggt/tokens2.npy')
    # print("Token2 shape: ", token2.shape)

    features = np.concatenate([token1, token2], axis=0)

    features = torch.from_numpy(features).float().cuda()

    for feature_idx in range(features.shape[0]):
        feature = features[feature_idx:feature_idx+1]
        retrieval_inds = retrieval_database.update(
            feature,
            add_after_query=True,
        )
        print("Frame: ", feature_idx)
        print("Retrieval inds: ", retrieval_inds)