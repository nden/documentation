General Notes
+++++++++++++

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
++++++++++++++++

- Fork the main jwst repository
- Create a local copy of the repository you just forked  and let it know where the main repository is.

```
git clone http://github.com/nden/jwst.git
git remote add jwst http://github.com/jwst/jwst.git
```

You should be able to look at all remotes now and see somthing like

```
% git remote -v
jwst   git://github.com/jwst/jwst.git (fetch)
jwst   git://github.com/jwst/jwst.git (push)
origin     git@github.com:your-user-name/jwst.git (fetch)
origin     git@github.com:your-user-name/jwst.git (push)
```

The above operations are normally performed once.
Now start work on a new feature/change by making a separate branch which tracks jwst/master.

First, always update the jwst/master branch to get the latest changes.
```
git fetch jwst
```

Make a new branch ``feature`` off jwst/master.

```
git checkout -b feature1 jwst/master
```

You will see a message like
```
Branch feature1 set to track jwst/master
```
- Work on this branch until the new feature is ready. 
  Then commit the changes
  ```
  git commit -m "Adding feature 1" file_names_to_commit
  ```
  and push to your forked repository
  ```
  git push origin feature`
  ```
  
  You can look at the changes online and if everyhting looks good 
  create a PR againts jwst/master. 
  
  - Add relevant labels and milestones and ask for a review.
  
  
