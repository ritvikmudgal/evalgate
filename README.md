# evalgate

Decide whether an eval delta is a real regression or just sampling noise, and
fail CI only when it is real.

```console
$ evalgate proportions --baseline-score 0.90 --baseline-n 2000 \
    --candidate-score 0.88 --candidate-n 2000
```

## License

MIT.
