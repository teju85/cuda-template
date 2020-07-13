#include <device_utils.cuh>
#include <catch2/catch.hpp>

namespace cuda {

TEST_CASE("ceil_div") {
  REQUIRE(ceil_div(3, 4) == 1);
  REQUIRE(ceil_div(4, 3) == 2);
  REQUIRE(ceil_div(4, 2) == 2);
}

TEST_CASE("align_to") {
  REQUIRE(align_to(3, 4) == 4);
  REQUIRE(align_to(4, 3) == 6);
  REQUIRE(align_to(4, 2) == 4);
}

TEST_CASE("align_down") {
  REQUIRE(align_down(3, 4) == 0);
  REQUIRE(align_down(4, 3) == 3);
  REQUIRE(align_down(4, 2) == 4);
}

TEST_CASE("is_po2") {
  REQUIRE(is_po2(2));
  REQUIRE(!is_po2(3));
}

TEST_CASE("log2") {
  REQUIRE(log2(4) == 2);
  REQUIRE(log2(7) == 2);
}

}  // namespace cuda
