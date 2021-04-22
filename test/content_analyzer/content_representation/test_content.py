import lzma
import pickle
from unittest import TestCase

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField, ContentField


class TestContent(TestCase):
    def test_load_serialize(self):
        features_bag = dict()
        features_bag["test_key"] = "test_value"
        content_field_repr = FeaturesBagField(features_bag)
        content_field = ContentField("0000")
        content_field.append(str(0), content_field_repr)
        content = Content("001")
        content.append("test_field", content_field)
        try:
            content.serialize(".")
        except FileNotFoundError:
            self.fail("Could not create file!")
        with lzma.open('001.xz', 'r') as file:
            self.assertEqual(content, pickle.load(file))

    def test_append_remove(self):
        features_bag = dict()
        features_bag["test_key"] = "test_value"
        content_field_repr = FeaturesBagField(features_bag)
        content_field = ContentField("0000")
        content_field.append(str(0), content_field_repr)
        content1 = Content("001")
        content1.append("test_field", content_field)

        content2 = Content("002")
        content2.append("test_field", content_field)
        content_field_repr = FeaturesBagField(features_bag)
        content_field2 = ContentField("0000")
        content_field2.append(str(0), content_field_repr)
        content2.append("test_field2", content_field2)
        content2.remove("test_field2")
        self.assertEqual(content1.field_dict, content2.field_dict)
