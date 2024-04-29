from detectron2.data import DatasetCatalog, MetadataCatalog
if "diseases_all_train" in DatasetCatalog.list():
    DatasetCatalog.remove("diseases_all_train")
    #DatasetCatalog.remove("leaf_all_test_")



from detectron2.data import DatasetCatalog, MetadataCatalog
if "diseases_all_test" in DatasetCatalog.list():
    #DatasetCatalog.remove("leaf_train_leaf_only_all_train")
    DatasetCatalog.remove("diseases_all_test")