#include <cudart_utils.h>
#include <catch2/catch.hpp>

namespace cuda {

TEST_CASE("THROW") {
  auto func = []() { THROW("This will throw an exception"); };
  REQUIRE_THROWS_AS(func(), std::runtime_error);
}

TEST_CASE("ASSERT") {
  auto func = [](bool pass) { ASSERT(pass, "This will try to assert"); };
  REQUIRE_THROWS_AS(func(false), std::runtime_error);
  REQUIRE_NOTHROW(func(true));
}

TEST_CASE("sync") {
  REQUIRE_NOTHROW(sync(0));
}

}  // namespace cuda
