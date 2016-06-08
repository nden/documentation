General Notes
-------------

All changes to the repository are made in branches on developer's fork of this rerpository
and submitted via a pull request (PR).

Each PR should be labeled with a step label (e.g. a PR to refpix should have a label ``refpix``)
and any other relevant labels (e.g. requirement number or Trac ticket number in case we are referencing those).
Each PR should be assigned a milestone (a build number or a future release version number).

A PR is merged after someone other than the submitter had looked at it. Usually this is one
of the two step maintainers. If nothing else simply adding a ``LGTM`` comment should be present.

Travis tests will run on the repository and PRs performing as a minimum a PEP8 check.
Steps which depend on other steps should have some sanity unit tests if possible as well.

Example Workflow
----------------

