# !/usr/bin/python3
# _*_coding: utf-8_*_

import os

import tensorflow as tf

import reader


class LmReaderTest(tf.test.TestCase):

    def setUp(self):
        self._string_data = '\n'.join(
            ["hello there i am",
             " rain as day",
             " want some cheesy puffs?"])

    def testRawData(self):
        tmpdir = tf.test.get_temp_dir()
        filename = os.path.join(tmpdir, "ptb.%s.txt" % "train")
        with tf.gfile.GFile(filename, 'w') as fh:
            fh.write(self._string_data)
        output = reader.raw_data(tmpdir)
        self.assertEqual(len(output), 2)

    def testDataProducer(self):
        raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
        batch_size = 3
        num_steps = 2
        x, y = reader.data_producer(raw_data, batch_size, num_steps)
        with self.test_session() as session:
            coord = tf.train.Coordinator()

            tf.train.start_queue_runners(session, coord=coord)
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
                self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
                self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
            finally:
                coord.request_stop()
                coord.join()




if __name__ == "__main__":
    tf.test.main()
