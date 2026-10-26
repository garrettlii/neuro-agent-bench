import pytest

from conftest import git
from nabench.prereg import PreregError, check_prereg

PREREG = """# Pre-registration

```yaml
plan:
  seed: 1
  blocks: [{{tasks: [dummy_spikes], conditions: [C1], models: [fake-scripted]}}]
predictions:
  min_normalized: {top}
```
"""


def test_prereg_guard(tmp_path):
    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    repo.mkdir()
    git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    git(repo, "init", "-b", "main")
    f = repo / "prereg.md"
    f.write_text(PREREG.format(top="<<FILL: number>>"))
    with pytest.raises(PreregError, match="placeholder"):
        check_prereg("prereg.md", repo=repo)
    f.write_text(PREREG.format(top="0.1"))
    with pytest.raises(PreregError):
        check_prereg("prereg.md", repo=repo)  # untracked
    git(repo, "add", "prereg.md")
    git(repo, "commit", "-m", "Add pre-registration")
    with pytest.raises(PreregError, match="remote"):
        check_prereg("prereg.md", repo=repo)
    info = check_prereg("prereg.md", repo=repo, require_pushed=False)
    assert info["predictions"]["min_normalized"] == 0.1
    assert info["plan"]["blocks"][0]["tasks"] == ["dummy_spikes"]
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "main")
    info = check_prereg("prereg.md", repo=repo)
    assert info["pushed_to"] == ["origin/main"]
    f.write_text(PREREG.format(top="0.2"))
    with pytest.raises(PreregError, match="uncommitted"):
        check_prereg("prereg.md", repo=repo)


@pytest.mark.parametrize("folder", ["config", "tasks"])
def test_prereg_guard_requires_clean_config_and_tasks(tmp_path, folder):
    repo = tmp_path / "repo"
    (repo / folder).mkdir(parents=True)
    git(repo, "init", "-b", "main")
    (repo / "prereg.md").write_text(PREREG.format(top="0.1"))
    tracked = repo / folder / "file.yaml"
    tracked.write_text("a: 1\n")
    git(repo, "add", "prereg.md", f"{folder}/file.yaml")
    git(repo, "commit", "-m", "Add pre-registration")
    check_prereg("prereg.md", repo=repo, require_pushed=False)
    tracked.write_text("a: 2\n")
    with pytest.raises(PreregError, match=f"{folder}/"):
        check_prereg("prereg.md", repo=repo, require_pushed=False)  # modified tracked file
    git(repo, "checkout", "--", f"{folder}/file.yaml")
    (repo / folder / "new.txt").write_text("x\n")
    with pytest.raises(PreregError, match=f"{folder}/"):
        check_prereg("prereg.md", repo=repo, require_pushed=False)  # new untracked file


def test_prereg_needs_plan_and_predictions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / "prereg.md").write_text("```yaml\npredictions: {a: 1}\n```\n")
    git(repo, "add", "prereg.md")
    git(repo, "commit", "-m", "Add pre-registration")
    with pytest.raises(PreregError, match="plan"):
        check_prereg("prereg.md", repo=repo, require_pushed=False)
