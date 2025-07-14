#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="/opt/ml/input/data/train",
        edge_paths=[
            "/opt/ml/input/data/train/freebase_mtr100_mte100-train_partitioned",
            "/opt/ml/input/data/train/freebase_mtr100_mte100-valid_partitioned",
            "/opt/ml/input/data/train/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="/opt/ml/model/",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=400,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=3,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
        num_gpus=1,
    )

    return config
