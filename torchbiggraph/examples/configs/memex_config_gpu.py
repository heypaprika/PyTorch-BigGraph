#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="data/memex/latest_processing",
        edge_paths=[
            "data/memex/latest_processing/edge-train_partitioned",
            "data/memex/latest_processing/edge-valid_partitioned",
            "data/memex/latest_processing/edge-test_partitioned",
        ],
        checkpoint_path="model/memex/letest_processing",
        # Graph structure
        entities = {
            "user": {"num_partitions": 1},
            "content": {"num_partitions": 1},
            "image": {"num_partitions": 1},
            "hashtag": {"num_partitions": 1},
        },
        relations = [
            {"name": "follows", "lhs": "user", "rhs": "user", "operator": "diagonal"},
            {"name": "likes", "lhs": "user", "rhs": "content", "operator": "diagonal"},
            {"name": "writes", "lhs": "user", "rhs": "content", "operator": "diagonal"},
            {"name": "has_image", "lhs": "content", "rhs": "image", "operator": "diagonal"},
            {"name": "tagged_with", "lhs": "content", "rhs": "hashtag", "operator": "diagonal"},
            {"name": "reposts", "lhs": "user", "rhs": "content", "operator": "diagonal"},
            {"name": "replies", "lhs": "user", "rhs": "content", "operator": "diagonal"},
        ],

        dynamic_relations=False,
        # Scoring model
        dimension=400,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=50,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
        workers=2,
        num_gpus=1
    )

    return config
