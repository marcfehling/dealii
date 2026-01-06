#!/bin/sh
## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2025 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------

#
# This is a little script that checks if the feature branch is linear,
# i.e., no merges of the parent branch into the feature branch are present.
#
# The feature branch as well as the parent branch to check against can
# be supplied as parameters to this script, or otherwise they will be set
# to 'HEAD' and 'master' by default, respectively.
#
# NOTE: This script will do nothing if feature and parent branch are the
#       same since the log will always be empty.
#

feature_ref="${1:-HEAD}"
parent_ref="${2:-master}"

get_merge_commits_since_parent_branch () {
  echo "$(git log --merges --pretty=format:"%h" ${parent_ref}..${feature_ref})"
}

get_commit_parents () {
  result=$(git rev-list --parents -n1 $1)
  # first result is the commit itself, which we omit
  echo "$(cut --delimiter=' '  --fields=1 --complement <<< ${result})"
}

commit_is_in_parent_branch () {
  return $(git merge-base --is-ancestor $1 ${parent_ref})
}

readarray -t merge_hash_array < <(get_merge_commits_since_parent_branch)

if [[ -z "${merge_hash_array}" ]] ; then
    echo "No merge commits present at all, everything is good!"
    exit 0
fi

echo "Merge commits found, checking if they originate from ${parent_ref}."

for hash in "${merge_hash_array[@]}"; do
  readarray -d' ' merge_parents < <(get_commit_parents ${hash})

  for parent_hash in "${merge_parents[@]}"; do
    if (commit_is_in_parent_branch ${parent_hash}) ; then
      echo "There is a merge commit coming from ${parent_ref}!"
      echo "The commit hash is ${hash}"
      exit 1
    fi
  done
done

echo "None do, everything is good!"
exit 0
